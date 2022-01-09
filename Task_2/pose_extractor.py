import cv2
from argparse import ArgumentParser
import mediapipe as mp
import numpy as np
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

mp_dict = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}


def parse_video(video_path):
    pose_landmarks = []
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    timestamps = []

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            time = float(frame_count) / fps
            timestamps.append(time)
            frame_count += 1
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            pose_landmarks.append(results.pose_landmarks)

    cap.release()
    cv2.destroyAllWindows()

    return pose_landmarks, timestamps


def get_landmarks_from_ind(frame, landmark_index):
    return {'x': frame.landmark[landmark_index].x, 'y': frame.landmark[landmark_index].y,
            'z': frame.landmark[landmark_index].z}


def get_all_landmarks(pose_landmarks):
    all_landmarks = []
    for frame in pose_landmarks:
        cur_pose = {}

        for point_name, point_ind in mp_dict.items():
            cur_pose[point_name] = get_landmarks_from_ind(frame, point_ind)

        all_landmarks.append(cur_pose)

    return all_landmarks


def get_get_average_points(left_points, right_points):
    cur_point = {}
    for key in left_points:
        cur_point[key] = {}
        cur_point[key]['x'] = (left_points[key]['x'] + right_points[key]['x']) / 2
        cur_point[key]['y'] = (left_points[key]['y'] + right_points[key]['y']) / 2
        cur_point[key]['z'] = (left_points[key]['z'] + right_points[key]['z']) / 2

    return cur_point


def smoothing(all_landmarks):
    for i in range(len(all_landmarks)):
        if len(all_landmarks[i]) != 33:
            right_neighbors = [j for j in range(i + 1, len(all_landmarks)) if len(all_landmarks[j]) == 33]
            right_neighbor = min(right_neighbors) if len(right_neighbors) > 0 else None

            left_neighbors = [j for j in range(i + 1, len(all_landmarks)) if len(all_landmarks[j]) == 33]
            left_neighbor = min(left_neighbors) if len(left_neighbors) > 0 else None

            if right_neighbor is None:
                all_landmarks[i] = all_landmarks[left_neighbor]
            elif left_neighbor is None:
                all_landmarks[i] = all_landmarks[right_neighbor]
            else:
                all_landmarks[i] = get_get_average_points(all_landmarks[left_neighbor], all_landmarks[right_neighbor])


def add_dummy_vertices(all_landmarks):
    for i in range(len(all_landmarks)):
        all_landmarks[i]['neck'] = {
            'x': (all_landmarks[i]['left_shoulder']['x'] + all_landmarks[i]['right_shoulder']['x']) / 2,
            'y': (all_landmarks[i]['left_shoulder']['y'] + all_landmarks[i]['right_shoulder']['y']) / 2,
            'z': (all_landmarks[i]['left_shoulder']['z'] + all_landmarks[i]['right_shoulder']['z']) / 2
        }

        all_landmarks[i]['backbone_end'] = {
            'x': (all_landmarks[i]['left_hip']['x'] + all_landmarks[i]['right_hip']['x']) / 2,
            'y': (all_landmarks[i]['left_hip']['y'] + all_landmarks[i]['right_hip']['y']) / 2,
            'z': (all_landmarks[i]['left_hip']['z'] + all_landmarks[i]['right_hip']['z']) / 2
        }


def get_tree_connections():
    connections = set(mp_pose.POSE_CONNECTIONS)
    connections.discard((29, 31))
    connections.discard((30, 32))
    connections.discard((17, 19))
    connections.discard((18, 20))
    connections.discard((12, 24))
    connections.discard((11, 23))
    connections.discard((11, 12))
    connections.discard((23, 24))
    connections.discard((9, 10))

    connections.add((33, 12))
    connections.add((33, 11))
    connections.add((33, 34))
    connections.add((34, 24))
    connections.add((34, 23))

    connections.add((33, 0))

    return connections


def get_tree_reversed_mp_dict():
    new_mp_dict = {**mp_dict, **{'neck': 33,
                                 'backbone_end': 34}}

    return {v: k for k, v in new_mp_dict.items()}


def get_angular_coordinate(x, y, z):
    theta, phi = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2)), np.arctan(y / x)

    if x < 0:
        phi += np.pi if y >= 0 else (-np.pi)

    return theta, phi


def get_all_landmarks_angular_coordinate(all_landmarks):
    all_landmarks_angular_coordinate = []

    connections = get_tree_connections()
    reversed_mp_dict = get_tree_reversed_mp_dict()

    for i in range(len(all_landmarks)):
        all_landmarks_angular_coordinate.append({})
        for connection in connections:
            landmark_1, landmark_2 = reversed_mp_dict[connection[0]], reversed_mp_dict[connection[1]]

            x = all_landmarks[i][landmark_2]['x'] - all_landmarks[i][landmark_1]['x']
            y = all_landmarks[i][landmark_2]['y'] - all_landmarks[i][landmark_1]['y']
            z = all_landmarks[i][landmark_2]['z'] - all_landmarks[i][landmark_1]['z']

            all_landmarks_angular_coordinate[-1][landmark_2] = get_angular_coordinate(x, y, z)

    return all_landmarks_angular_coordinate


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--src', default='panoptic_sample.mp4', help='Path to video')
    arg_parser.add_argument('--dst', default='result.csv', help='Path to csv file with landmarks')
    args = arg_parser.parse_args()

    pose_landmarks, timestamps = parse_video(args.src)
    all_landmarks = get_all_landmarks(pose_landmarks)
    smoothing(all_landmarks)
    add_dummy_vertices(all_landmarks)
    all_landmarks_angular_coordinate = get_all_landmarks_angular_coordinate(all_landmarks)

    landmarks = pd.DataFrame(all_landmarks_angular_coordinate)
    landmarks['time'] = ['{0:01.0f}:{1:04.0f}'.format(*divmod((time) * 60, 60)) for time in timestamps]
    landmarks = landmarks.set_index('time')
    landmarks.to_csv(args.dst)
