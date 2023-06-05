import argparse
import os
import os.path as osp
import time
import schedule
import cv2
import torch
import logging
import requests
import json

from loguru import logger
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from aga_engine.aga_predictor import AGAPredictor




def make_parser():
    # 인자 파서 생성
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    # demo 유형 설정
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )

    # 실험 이름 설정
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)

    # 모델 이름 설정
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # 이미지 또는 비디오 경로 설정
    parser.add_argument(
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )

    # 웹캠 사용 시 카메라 ID 설정
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

    # 이미지 또는 비디오 결과 저장 여부 설정
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # 실험 파일 및 가중치 파일 설정
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")

    # 모델 실행 디바이스 설정
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    # 테스트 설정 (confidence, nms, image size)
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")

    # 프레임 속도 설정
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")

    # 모델 실행 설정 (fp16, fuse, TensorRT)
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )

    # 추적 관련 설정
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=0, help='the min box area for tracking.')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser



class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model  # 모델 객체
        self.decoder = decoder  # 디코더 객체
        self.num_classes = exp.num_classes  # 클래스 수
        self.confthre = exp.test_conf  # 추론 시 사용할 신뢰 임계값
        self.nmsthre = exp.nmsthre  # 추론 시 사용할 NMS(non-maximum suppression) 임계값
        self.test_size = exp.test_size  # 테스트 이미지 크기
        self.device = device  # 디바이스 (CPU or GPU)
        self.fp16 = fp16  # FP16 형식 사용 여부
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)  # 이미지 RGB 평균값
        self.std = (0.229, 0.224, 0.225)  # 이미지 표준편차

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # FP16 형식으로 변환

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)  # 모델에 이미지를 전달하여 출력 얻기
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )  # 추론 결과 후처리 수행 (클래스 필터링 및 NMS 적용)
        return outputs, img_info


def imageflow_demo(predictor, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)  # 비디오 경로 또는 카메라 ID로 비디오 캡처 객체 생성
    tracker = BYTETracker(args, frame_rate=30)  # BYTETracker 객체 생성
    aga_pred = AGAPredictor(logger=logging, is_debug=True)  # AGAPredictor 객체 생성
    aga_pred.load_model()  # AGAPredictor 모델 로드
    timer = Timer()  # 타이머 객체 생성
    frame_id = 0  # 프레임 ID 초기화
    data_list = [] # tid당 성별 평균을 구하기위한 리스트
    stored_data = []
    
    def add_data_to_list(tid, gender, front, side):
        data_time = time.strftime('%Y-%m-%d %H:%M:%S')
        data = {
            'tid': tid,
            'gender_sum': gender,
            'gender_count': 1,
            'front': front,
            'frontcount' : 0,
            'side': side,
            'data_time': data_time
        }
        data_list.append(data)

    def calculate_effect():
        front_count = 0
        for data in data_list:
            front = data['front']
            side = data['side']
            if front_count >= 4:
                return 2
            elif side >= 0.4 and front >= 0.2:
                return 1
            else:
                return 0
                
        
    
    def send_data():
        data = []
        for item in data_list:
            if((item['gender_sum'] / item['gender_count']) >= 0.25):
                result_gender = 'W'
            else:
                result_gender = 'M'
            
            item_data = {
                'data_time': str(item['data_time']),
                'effect': str(calculate_effect()),
                'gender': str(result_gender)
            }
            data.append(item_data)
            
        dumps_data = json.dumps(data)
        json_data = {
            'tid_data': dumps_data
        }
        
        url = "http://211.223.37.186:9000/submit"
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(url, data=json_data, headers=headers)
        if response.status_code == 200:
            print("데이터 전송 성공")
            data_list.clear()
        else:
            print("데이터 전송 실패 (상태 코드: {})".format(response.status_code))
            stored_data.extend(data_list)
            data_list.clear()
    

    schedule.every(5).minutes.do(send_data)
    
    
    while True:
        
        if frame_id % 30 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            schedule.run_pending()
        ret_val, frame = cap.read()  # 비디오에서 프레임 읽기
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)  # 프레임을 모델에 전달하여 추론 수행
            
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)  # 객체 추적 업데이트 수행
                for t in online_targets:
                    tlwh = t.tlwh  # 객체의 Top-Left Width-Height 좌표 가져오기
                    tid = t.track_id  # 객체의 고유 ID 가져오기 
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh  # 객체의 가로 세로 비율이 임계값보다 큰지 확인
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:  # 객체의 면적과 가로 세로 비율이 임계값을 만족하는지 확인
                        x0 = int(tlwh[0])  # 객체의 왼쪽 상단 X 좌표
                        y0 = int(tlwh[1])  # 객체의 왼쪽 상단 Y 좌표
                        x1 = int(tlwh[0] + tlwh[2])  # 객체의 오른쪽 하단 X 좌표
                        y1 = int(tlwh[1] + tlwh[3])  # 객체의 오른쪽 하단 Y 좌표
                        if frame_id % 5 == 0: # 5프레임에 한번씩 속성 분석 수행
                            if x0 >= 0 and y0 >= 0 and x1 <= frame.shape[1] and y1 <= frame.shape[0]:  # 객체가 프레임 내에 있는지 확인
                                img_crop = frame[y0:y1, x0:x1, :]  # 객체 영역을 잘라내기
                                valid_props, attns = aga_pred.infer_batch_once([img_crop])  # 잘라낸 객체 영역에 대해 속성 분석 수행
                                gender = valid_props[0][22]
                                front = valid_props[0][23]
                                side = valid_props[0][24]
                                for data in data_list:
                                    if data['tid'] == tid:
                                        data['gender_sum'] += gender
                                        data['gender_count'] += 1
                                        data['front'] = max(data['front'], front)
                                        data['frontcount'] += (1 if front >= 0.6 else 0)
                                        data['side'] = max(data['side'], side)
                                        data['effect'] = calculate_effect()
                                        break
                                else:
                                    add_data_to_list(tid, gender, front, side)

                                
                                # 현재 시간, 객체 ID, 성별 및 주목도 정보 출력
                                logger.info(f"time:{time.strftime('%Y-%m-%d %H:%M:%S')}\ttid:{tid}\tgender:{valid_props[0][22]:.3f}\t Front:{valid_props[0][23]:.3f}\t side:{valid_props[0][24]:.3f}\tback:{valid_props[0][25]:.3f}")  
                timer.toc()  # 타이머 종료
            else:
                timer.toc()  # 타이머 종료
            cv2.imshow("track",frame)  # 추적된 객체가 표시된 프레임 표시
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):  # 'q' 또는 'Q' 키를 누르면 종료
                break
        else:
            break
        frame_id += 1  # 다음 프레임으로 이동
    cap.release()  # 비디오 캡처 객체 해제
    cv2.destroyAllWindows()  # 창 닫기

    

def main(exp, args):
    logger.info("start")  # 시작 로그 출력
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)  # 출력 디렉토리 경로 생성
    os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리 생성 (이미 존재하는 경우 덮어쓰지 않음)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")  # 결과 저장 폴더 경로 생성
        os.makedirs(vis_folder, exist_ok=True)  # 결과 저장 폴더 생성 (이미 존재하는 경우 덮어쓰지 않음)

    if args.trt:
        args.device = "gpu"  # TensorRT 사용 시 디바이스를 GPU로 설정
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")  # 디바이스 설정

    if args.conf is not None:
        exp.test_conf = args.conf  # 테스트 시의 confidence 임계값 설정
    if args.nms is not None:
        exp.nmsthre = args.nms  # 테스트 시의 NMS 임계값 설정
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)  # 테스트 시의 입력 이미지 크기 설정

    model = exp.get_model().to(args.device)  # 모델 생성 및 디바이스 설정
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))  # 모델 요약 정보 출력
    model.eval()  # 모델을 평가 모드로 설정

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")  # 체크포인트 파일 경로 생성
        else:
            ckpt_file = args.ckpt  # 사용자가 지정한 체크포인트 파일 경로 사용
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")  # 체크포인트 파일 로드
        # load the model state dict
        model.load_state_dict(ckpt["model"])  # 모델의 상태 사전을 로드
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)  # 모델 퓨전 수행

    if args.fp16:
        model = model.half()  # FP16 형식으로 모델을 변환

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)  # Predictor 객체 생성
    imageflow_demo(predictor, args)  # 이미지 플로우 데모 실행
    logger.info("finish")  # 완료 로그 출력

if __name__ == "__main__":
    args = make_parser().parse_args()  # 명령줄 인수 파싱
    exp = get_exp(args.exp_file, args.name)  # 실험 설정 가져오기

    main(exp, args)  # 메인 함수 실행

