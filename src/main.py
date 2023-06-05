from pathlib import Path
from typing import Tuple, Optional, Union, Dict
from dataclasses import dataclass

import numpy as np
import cv2

import tracker
from detector import Detector
from utils.logging_utils_local import get_logger

logger = get_logger(name='traffic_monitoring_system_simple')
base_path = Path(__file__).resolve().parent


def get_cv2_video_writer(video_save_path: str, saved_fps: float,
                         frame_size: Tuple[int, int] = (1920, 1080), save_color: bool = True) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MP4 格式
    out = cv2.VideoWriter(video_save_path, fourcc=fourcc, fps=saved_fps, frameSize=frame_size, isColor=save_color)
    print(f"Video Writer initialized finished, video_save_path={video_save_path}, saved_fps={saved_fps}, "
          f"frame_size={frame_size}, save_color={save_color}")
    return out


@dataclass
class TargetPositionInfo:
    x: int = 0
    y: int = 0
    time_step: int = 0
    id: Optional[int] = None


class CustomFixedLengthArray(list):
    """maybe success from list is unnecessary, only use its __getitem__ method """

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
        :param element:
        :return:
        """
        if self.front_pointer_index == self.end_next_pointer_index and not self.array_empty_flag:
            # array is full
            return False, "array is full"

        self.end_next_pointer_index = (self.end_next_pointer_index + 1) % self.array_len
        self.front_pointer_index = (self.front_pointer_index + 1) % self.array_len
        self.array_list[self.end_next_pointer_index] = element
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

    def get_history_distance(self) -> Tuple[bool, Union[str, Tuple[float, int]]]:
        """iterate over all elements in array: if elements only have <=1 history pos, skip it. Else calculate the distance"""
        pixel_distance: float = 0.0
        total_frame_step: int = 0
        if self.array_empty_flag or (self.front_pointer_index + 1) % self.array_len == self.end_next_pointer_index:
            return False, "empty array or only 1 element"

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


if __name__ == '__main__':
    video_filepath = base_path.joinpath("video/demo.mov")
    video_process_result_save_path = base_path.joinpath(
        f"video/{video_filepath.stem}_process_result.avi")  # correspond with VideoWriter_fourcc encoding type
    assert video_filepath.exists(), f"{video_filepath.as_posix()} video not exists!"

    save_frame_size: Tuple[int, int] = (960, 540)  # frame resize
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)

    # 初始化2个撞线polygon
    list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
                     [299, 375], [267, 289]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
                       [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, save_frame_size)

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, save_frame_size)

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(save_frame_size[0] * 0.01), int(save_frame_size[1] * 0.05))

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture(video_filepath.as_posix())
    # capture = cv2.VideoCapture('/mnt/datasets/datasets/towncentre/TownCentreXVID.avi')
    # get video parameters
    video_save_flag: bool = capture.isOpened()
    video_writer: Optional[cv2.VideoWriter] = None

    frame_rate: float = 25.
    if video_save_flag:
        frame_total_num = capture.get(7)
        frame_rate = float(capture.get(5))  # FPS
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer: Optional[cv2.VideoWriter] = get_cv2_video_writer(
            video_save_path=video_process_result_save_path.as_posix(),
            saved_fps=frame_rate,
            frame_size=save_frame_size,  # (int(frame_width),int(frame_height)),
            save_color=True, )
    try:
        vehicle_speed_id_map_array: Dict[int, CustomFixedLengthArray] = {}
        current_frame_index_in_video: int = 0
        while True:
            # 读取每帧图片
            _, im = capture.read()
            if im is None:
                break

            # 缩小尺寸，1920x1080->960x540
            im = cv2.resize(im, save_frame_size)

            list_bboxs = []
            bboxes = detector.detect(im)

            # 如果画面中 有bbox
            if len(bboxes) > 0:
                list_bboxs = tracker.update(bboxes, im)

                # 画框
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None,
                                                         current_frame_index_in_video=current_frame_index_in_video,
                                                         video_fps=frame_rate)
                pass
            else:
                # 如果画面中 没有bbox
                output_image_frame = im
            pass

            # 输出图片
            output_image_frame = cv2.add(output_image_frame, color_polygons_image)

            if len(list_bboxs) > 0:
                # ----------------------判断撞线----------------------
                for item_bbox in list_bboxs:
                    x1, y1, x2, y2, label, track_id = item_bbox

                    # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                    y1_offset = int(y1 + ((y2 - y1) * 0.6))

                    # 撞线的点
                    y = y1_offset
                    x = x1

                    if polygon_mask_blue_and_yellow[y, x] == 1:
                        # 如果撞 蓝polygon
                        if track_id not in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.append(track_id)
                        pass

                        # 判断 黄polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 外出方向
                        if track_id in list_overlapping_yellow_polygon:
                            # 外出+1
                            up_count += 1

                            print(
                                f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_yellow_polygon}')

                            # 删除 黄polygon list 中的此id
                            list_overlapping_yellow_polygon.remove(track_id)

                            pass
                        else:
                            # 无此 track_id，不做其他操作
                            pass

                    elif polygon_mask_blue_and_yellow[y, x] == 2:
                        # 如果撞 黄polygon
                        if track_id not in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.append(track_id)
                        pass

                        # 判断 蓝polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 进入方向
                        if track_id in list_overlapping_blue_polygon:
                            # 进入+1
                            down_count += 1

                            print(
                                f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')

                            # 删除 蓝polygon list 中的此id
                            list_overlapping_blue_polygon.remove(track_id)

                            pass
                        else:
                            # 无此 track_id，不做其他操作
                            pass
                        pass
                    else:
                        pass
                    pass

                pass

                # ----------------------清除无用id----------------------
                list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
                for id1 in list_overlapping_all:
                    is_found = False
                    for _, _, _, _, _, bbox_id in list_bboxs:
                        if bbox_id == id1:
                            is_found = True
                            break
                        pass
                    pass

                    if not is_found:
                        # 如果没找到，删除id
                        if id1 in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.remove(id1)
                        pass
                        if id1 in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.remove(id1)
                        pass
                    pass
                list_overlapping_all.clear()
                pass

                # 清空list
                list_bboxs.clear()

                pass
            else:
                # 如果图像中没有任何的bbox，则清空list
                list_overlapping_blue_polygon.clear()
                list_overlapping_yellow_polygon.clear()
                pass
            pass

            text_draw = 'DOWN: ' + str(down_count) + \
                        ' , UP: ' + str(up_count)
            output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                             org=draw_text_postion,
                                             fontFace=font_draw_number,
                                             fontScale=1, color=(255, 255, 255), thickness=2)

            cv2.imshow('demo', output_image_frame)
            if video_save_flag:
                video_writer.write(output_image_frame)
            cv2.waitKey(1)

            pass
            current_frame_index_in_video += 1
    except Exception as e:
        logger.exception(f"error={e}")

    capture.release()
    video_writer.release()
    cv2.destroyAllWindows()
