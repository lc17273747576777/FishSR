import math
import cv2
import numpy as np
import math
import os

# 定义FisheyeIm类
class FisheyeIm():
    # 初始化内参和畸变系数
    def __init__(self, K, D):
        self.K = K
        self.D = D
        self.KS = K.I

    # 求解角度畸变方程
    def solvePhi(self, phif):
        phi = phif
        f = 1
        num = 0
        while(abs(f) > 0.0001 and abs(f) < 100):
            phi2 = phi*phi
            phi4 = phi2*phi2
            phi6 = phi4*phi2
            phi8 = phi4*phi4

            f = phi*(1 + self.D[0, 0] * phi2 +
                     self.D[0, 1] * phi4 + self.D[0, 2] * phi6 + self.D[0, 3] * phi8)-phif
            f_ = 1 + 3 * self.D[0, 0] * phi2 + 5 * self.D[0, 1] * \
                phi4 + 7 * self.D[0, 2] * phi6 + 9 * self.D[0, 3] * phi8
            phi = phi-f/f_
            num = num+1
            if num > 30:
                break
        if num > 29 or abs(phi-phif) > 1:
            return -1
        else:
            return phi

    # 将input转换为鱼眼图像
    # size_fisheye模拟鱼眼图像的大小
    def GetFisheyeMap(self, size_fisheye):
        (hf, wf) = size_fisheye
        mapX = np.mat(np.zeros((hf, wf)))
        mapY = np.mat(np.zeros((hf, wf)))
        # print("正在初始化")
        for y in range(hf):
            for x in range(wf):
                x_ = self.KS[0, 0]*x+self.KS[0, 1]*y+self.KS[0, 2]
                y_ = self.KS[1, 0]*x+self.KS[1, 1]*y+self.KS[1, 2]
                phif = math.sqrt(x_**2+y_**2)
                theta = math.atan2(y_, x_)
                phi = self.solvePhi(phif)
                rs = math.tan(phi)
                if rs > 0:
                    x_ = rs*math.cos(theta)
                    y_ = rs*math.sin(theta)
                    mapX[y, x] = self.K[0, 0]*x_ + self.K[0, 1]*y_
                    mapY[y, x] = self.K[1, 0]*x_ + self.K[1, 1]*y_
                else:
                    mapX[y, x] = -1.
                    mapY[y, x] = -1.
        self.mapX = mapX
        self.mapY = mapY
        # print("初始化完成")

    # input:输入图像
    # k:原图放大倍数
    def ChangeToFisheye(self, input, k):

        nmapX = self.mapX+self.K[0, 2]*k
        nmapY = self.mapY+self.K[1, 2]*k
        nmapX = np.where(self.mapX != -1., self.mapX+self.K[0, 2]*k, -1)
        nmapY = np.where(self.mapY != -1., self.mapY+self.K[1, 2]*k, -1)

        (hf, wf, _) = input.shape
        h = round(hf*k)
        w = round(wf*k)
        input = cv2.resize(input, (w, h))
        output = cv2.remap(input, np.float32(nmapX), np.float32(
            nmapY), interpolation=cv2.INTER_LINEAR)
        return output


class Distort_restore():
    # 构造函数，初始化摄像头
    # number:摄像头编号，width:图片宽度，height:图片高度
    def __init__(self, input_img, width, height, ):
        self.width = width
        self.height = height

        self.input_img = input_img
    # 用于拍摄，拍摄结果存入列表image中
    def ShowCamera_single(self):
        DIM = (self.width, self.height)
        K = np.array([[600.0539928016384, 0, 960.2618818766359], [
            0, 600.2506744146734, 519.4571581216711], [0, 0, 1]])
        D = np.array([[-0.024092199861108887], [0.002745976275100771],
                      [0.002545415522352827], [-0.0014366825722748522]])
        # D = np.array([[-2.0678964965317591e-02, -1.9451353233073934e-03,
        #                 1.1199276330185410e-03, -5.3007052743606401e-04]])
        Knew = K.copy()
        scale = 1.0
        Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
        order = 'z'
        img = self.input_img
        undistorted_img = cv2.remap(
            img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # cv2.imwrite(osp.join(self.output_path, img_name) + '_restore.jpg', undistorted_img)
        return undistorted_img

# 示例
#初始化部分，确定输出图像尺寸后只运行一次
K = np.matrix([[600.05399280163840, 0, 960.26188187663593],
               [0, 600.25067441467343, 519.45715812167111],
               [0, 0, 1]])

D = np.matrix([[-0.024092199861108887, 0.002745976275100771,
               0.002545415522352827, -0.0014366825722748522]])

# D = np.matrix([[-2.0678964965317591e-02, -1.9451353233073934e-03,
#                1.1199276330185410e-03, -5.3007052743606401e-04]])

# input_img_path = "F:/WJ_project/dataset/realSR/ori/Flickr2K_HR"
input_img_path = "F:/WJ_project/dataset/realSR/ori/DIV2K_train_HR"
start_num = 0
for dirpath, dirnames, filenames in os.walk(input_img_path):
    for filename in filenames:
        if start_num > 1:
            start_num -= 1
            continue
        print(filename)
        total_path = os.path.join(dirpath, filename)
        input_img = cv2.imread(total_path)
        input_img = cv2.resize(input_img, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
        H, W, C = input_img.shape

        fisheyeim = FisheyeIm(K, D)  # 创建FisheyeIm，使用K,D初始化
        fisheyeim.GetFisheyeMap([H, W])  # 设定输出图片大小（与内参一致）计算mapX,mapY

        k = 1  # 放大倍数
        T = cv2.getTickCount()
        output = fisheyeim.ChangeToFisheye(input_img, k)

        output_path = "F:/WJ_project/dataset/realSR/ori/DIV2K_train_LR_2D"


        img_name = os.path.splitext(os.path.basename(input_img_path))[0]
        # img_name_fisheye = img_name + str(k) + '_fisheye.jpg'
        # save_img_name0 = os.path.join(output_path, img_name_fisheye)
        # cv2.imwrite(save_img_name0, output)
        # distort = Distort_restore(output, 1920, 1080)
        distort = Distort_restore(output, W, H)
        distort_result = distort.ShowCamera_single()
        file = filename.replace('.png', '')
        img_name_sim = file + '_sim.png'
        save_img_name = os.path.join(output_path, img_name_sim)
        print("模拟时间:", (cv2.getTickCount()-T)/cv2.getTickFrequency()*1000, "s")
        cv2.imwrite(save_img_name, distort_result)

