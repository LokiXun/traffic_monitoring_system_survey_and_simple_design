from typing import Tuple, Optional, Union, Dict
from dataclasses import dataclass

import cv2
import torch
import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

cfg = get_config()
cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


@dataclass
class TargetPositionInfo:
    x: int = 0
    y: int = 0
    time_step: int = 0
    id: Optional[int] = None


class CustomFixedLengthArray(list):
    """maybe success from list maybe unnecessary, only use its __getitem__ method"""

    def __init__(self, array_len: int = 10,
                 example_element: TargetPositionInfo = TargetPositionInfo(x=0, y=0, id=None),
                 array_name: str = "None"):
        """

        :param array_len: vehicle's id
        :param example_element:
        :param array_name:
        """
        self.array_name = array_name
        self.array_len = array_len

        self.front_pointer_index = 0  # front element index
        self.end_next_pointer_index = 0  # next element after the end index
        self.array_empty_flag: bool = True  # differentiate array is empty or full

        self.array_list = [example_element] * self.array_len
        super().__init__(self.array_list)

    def push(self, element: TargetPositionInfo) -> Tuple[bool, str]:
        """
        fixed length array. when push a new element, need to discard array's front element
        WARNING: A little bit different from array, when arry is full,
            we need to discard the front one becuase it was stale
        :param element:
        :return:
        """
        if self.front_pointer_index == self.end_next_pointer_index and not self.array_empty_flag:
            # array is full
            # return False, "array is full"
            self.front_pointer_index = (self.front_pointer_index + 1) % self.array_len

        self.array_list[self.end_next_pointer_index] = element
        self.end_next_pointer_index = (self.end_next_pointer_index + 1) % self.array_len
        self.array_empty_flag = False

        return True, ""

    def pop(self, *args, **kwargs) -> Tuple[bool, Union[TargetPositionInfo, str]]:

        if self.front_pointer_index == self.end_next_pointer_index and self.array_empty_flag:
            # array is empty
            return False, "empty array"

        pop_element: TargetPositionInfo = self.array_list[self.front_pointer_index]
        self.front_pointer_index = (self.front_pointer_index + 1) % self.array_len
        if self.front_pointer_index == self.end_next_pointer_index:
            # after pop, front==end means empty array
            self.array_empty_flag = True

        return True, pop_element

    def __len__(self):
        """get valid elements' num"""
        if self.array_empty_flag:
            return 0
        if self.front_pointer_index == self.end_next_pointer_index:
            return self.array_len

        return (self.end_next_pointer_index - self.front_pointer_index) % self.array_len

    @property
    def judge_array_empty(self):
        return self.array_empty_flag

    @staticmethod
    def calculate_2_position_distance(last_pos: TargetPositionInfo, current_pos: TargetPositionInfo):
        return np.linalg.norm(np.array([last_pos.x - current_pos.x, last_pos.y - current_pos.y]))

    def get_history_distance_timestep(self) -> Tuple[bool, Union[str, Tuple[float, int]]]:
        """get_history_distance_timestep by iterating over all elements in array
        if elements only have <=1 history pos, skip it. Else calculate the distance"""
        pixel_distance: float = 0.0
        total_frame_step: int = 0
        if self.array_empty_flag:
            return False, "empty array"
        if (self.front_pointer_index + 1) % self.array_len == self.end_next_pointer_index:
            return False, "element not enough to calculate speed, only 1 element"

        current_index = self.front_pointer_index
        last_position_info: TargetPositionInfo = self.array_list[current_index]
        current_index = (current_index + 1) % self.array_len
        while current_index != self.end_next_pointer_index:
            current_position_info: TargetPositionInfo = self.array_list[current_index]
            # calculate pixel distance
            current_pixel_distance: float = self.calculate_2_position_distance(last_pos=last_position_info,
                                                                               current_pos=current_position_info)
            total_frame_step += current_position_info.time_step - last_position_info.time_step
            pixel_distance += current_pixel_distance
            current_index = (current_index + 1) % self.array_len

        return True, (pixel_distance, total_frame_step)

    def clear_array(self):
        self.front_pointer_index = 0
        self.end_next_pointer_index = 0
        self.array_empty_flag = True


vehicle_velocity_id_map_Array: Dict[int, CustomFixedLengthArray] = dict()

pixel2reality_scale: Optional[float] = None
car_reality_length: float = 5.0  # meters


#  pixel2reality_scale: what length in reality for one pixel (unit=m/pixel)
#  TODO: roughly assume car's length= 5m and calculate the scale = 5m / car_box_length


def draw_bboxes(image, bboxes, line_thickness, current_frame_index_in_video: int, video_fps: float):
    """

    :param image:
    :param bboxes:
    :param line_thickness:
    :param current_frame_index_in_video:
    :param video_fps:
    :return:
    """
    global pixel2reality_scale, car_reality_length
    global vehicle_velocity_id_map_Array

    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1

    list_pts = []
    point_radius = 4

    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)

        # initialize the pixel_scale from the first car detected
        if pixel2reality_scale is None:
            # not initialized yet
            pixel2reality_scale = car_reality_length / max((x2 - x1), (y2 - y1))  # reality_length / pixel_length

        # 撞线的点
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        """calcutale velocity from history frames"""
        if pos_id not in vehicle_velocity_id_map_Array:
            vehicle_velocity_id_map_Array[pos_id]: CustomFixedLengthArray = CustomFixedLengthArray()

        vehicle_velocity_id_map_Array[pos_id].push(TargetPositionInfo(x=check_point_x, y=check_point_y,
                                                                      id=pos_id,
                                                                      time_step=current_frame_index_in_video))

        # calculate speed from history
        success_flag, velocity_result = vehicle_velocity_id_map_Array[pos_id].get_history_distance_timestep()
        velocity_in_reality: Optional[float] = None
        if success_flag:
            pixel_distance, history_time_step = velocity_result
            # equation: $Velocity = \frac{pixel\_length}{frame\_num * 1/FPS} * realtity\_scale$
            velocity_in_reality: float = (pixel_distance / history_time_step * video_fps *
                                          pixel2reality_scale) * 3.6  # km/h
        velocity_in_reality_str_show: str = f"V={velocity_in_reality:.1f}km/h" if velocity_in_reality else ""

        # draw boxes onto images and append calculated information
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled

        put_text_font_scale: float = 0.5  # line_thickness / 3
        cv2.putText(image, f'{cls_id} ID-{pos_id} {velocity_in_reality_str_show}', (c1[0], c1[1] - 2), 0,
                    put_text_font_scale, [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

        ndarray_pts = np.array(list_pts, np.int32)

        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))

        list_pts.clear()

    return image


def update(bboxes, image):
    bbox_xywh = []
    confs = []
    bboxes2draw = []

    if len(bboxes) > 0:
        for x1, y1, x2, y2, lbl, conf in bboxes:
            obj = [
                int((x1 + x2) * 0.5), int((y1 + y2) * 0.5),
                x2 - x1, y2 - y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, image)

        for x1, y1, x2, y2, track_id in list(outputs):
            # x1, y1, x2, y2, track_id = value
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            label = search_label(center_x=center_x, center_y=center_y,
                                 bboxes_xyxy=bboxes, max_dist_threshold=20.0)

            bboxes2draw.append((x1, y1, x2, y2, label, track_id))
        pass
    pass

    return bboxes2draw


def search_label(center_x, center_y, bboxes_xyxy, max_dist_threshold):
    """
    在 yolov5 的 bbox 中搜索中心点最接近的label
    :param center_x:
    :param center_y:
    :param bboxes_xyxy:
    :param max_dist_threshold:
    :return: 字符串
    """
    label = ''
    # min_label = ''
    min_dist = -1.0

    for x1, y1, x2, y2, lbl, conf in bboxes_xyxy:
        center_x2 = (x1 + x2) * 0.5
        center_y2 = (y1 + y2) * 0.5

        # 横纵距离都小于 max_dist
        min_x = abs(center_x2 - center_x)
        min_y = abs(center_y2 - center_y)

        if min_x < max_dist_threshold and min_y < max_dist_threshold:
            # 距离阈值，判断是否在允许误差范围内
            # 取 x, y 方向上的距离平均值
            avg_dist = (min_x + min_y) * 0.5
            if min_dist == -1.0:
                # 第一次赋值
                min_dist = avg_dist
                # 赋值label
                label = lbl
                pass
            else:
                # 若不是第一次，则距离小的优先
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    # label
                    label = lbl
                pass
            pass
        pass

    return label
