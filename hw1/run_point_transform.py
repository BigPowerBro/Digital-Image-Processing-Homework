import cv2
import numpy as np
import gradio as gr
import math

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.zeros(image.shape)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    Ps = np.array(source_pts).reshape(-1,2)
    Qs = np.array(target_pts).reshape(-1,2)
    ### FILL: 基于MLS or RBF 实现 image warping
    img = affine_deformation(img,Ps,Qs)
    warped_image = np.array(img)

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

def affine_deformation(image,source_pts,target_pts):

    width, height = image.shape[:2]
    pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
    pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T
    img_coordinate = np.swapaxes(np.array([pcth, pctw]), 1, 2).T
    W,A,Z = cal_waz(source_pts,img_coordinate,height,width)
    target_pts = source_pts * 2 - target_pts
    cita = cal_G(img_coordinate,source_pts,height,width)
    mapxy = np.swapaxes(
        np.float32(cal_fv(target_pts, W, A, Z, height, width,cita,img_coordinate)),
        0, 1)
    img = cv2.remap(image, mapxy[:, :, 0], mapxy[:, :, 1],borderValue=(255,255,255), interpolation=cv2.INTER_LINEAR)
    return img

def cal_waz(source_pts, img_coord, height, width):
    wi = np.reciprocal(
        np.power(np.linalg.norm(np.subtract(source_pts, img_coord.reshape(height, width, 1, 2)) + 0.000000001, axis=3), 2))
    
    p_star = np.divide(np.matmul(wi, source_pts), np.sum(wi, axis=2).reshape(height, width, 1))

    phat = np.subtract(source_pts, p_star.reshape(height, width, 1, 2))

    z1 = np.subtract(img_coord, p_star)
    z2 = np.repeat(np.swapaxes(np.array([z1[:, :, 1], -z1[:, :, 0]]), 1, 2).T.reshape(height, width, 1, 2, 1),
                   [source_pts.shape[0]], axis=2)

    z1 = np.repeat(z1.reshape(height, width, 1, 2, 1), [source_pts.shape[0]], axis=2)
    s1 = phat.reshape(height, width, source_pts.shape[0], 1, 2)
    s2 = np.concatenate((s1[:, :, :, :, 1], -s1[:, :, :, :, 0]), axis=3).reshape(height, width, source_pts.shape[0], 1, 2)

    a = np.matmul(s1, z1)
    b = np.matmul(s1, z2)
    c = np.matmul(s2, z1)
    d = np.matmul(s2, z2)

    ws = np.repeat(wi.reshape(height, width, source_pts.shape[0], 1), [4], axis=3)

    A = (ws * np.concatenate((a, b, c, d), axis=3).reshape(height, width, source_pts.shape[0], 4)).reshape(height, width,
                                                                                                   source_pts.shape[0], 2, 2)
    return wi, A, z1

def cal_G(img_coord, source_pts, height, width, thre = 0.7):

    max = np.max(source_pts, 0)
    min = np.min(source_pts, 0)

    length = np.max(max - min)

    # 计算控制区域中心
    # p_ = (max + min) // 2
    p_ = np.sum(source_pts,axis=0) // source_pts.shape[0]

    # 计算控制区域
    minx, miny = min - length
    maxx, maxy = max + length
    minx = minx if minx > 0 else 0
    miny = miny if miny > 0 else 0
    maxx = maxx if maxx < height else height
    maxy = maxy if maxy < width else width

    k1 =(p_ - [0,0])[1] / (p_ - [0,0])[0]
    k2 =(p_ - [height,0])[1] / (p_ - [height,0])[0]
    k4 =(p_ - [0,width])[1] / (p_ - [0,width])[0]
    k3 =(p_ - [height, width])[1] / (p_ - [height, width])[0]
    k = (np.subtract(p_, img_coord)[:, :, 1] / (np.subtract(p_, img_coord)[:, :, 0] + 0.000000000001)).reshape(height, width, 1)
    k = np.concatenate((img_coord, k), axis=2)

    k[:,:p_[1],0][(k[:,:p_[1],2] > k1) | (k[:,:p_[1],2] < k2)] = \
    (np.subtract(p_[1], k[:,:,1]) / p_[1]).reshape(height, width, 1)[:,:p_[1],0][(k[:,:p_[1],2] > k1) | (k[:,:p_[1],2] < k2)]
    k[:,p_[1]:,0][(k[:,p_[1]:,2] > k3) | (k[:,p_[1]:,2] < k4)] = \
    (np.subtract(k[:,:,1], p_[1]) / (width - p_[1])).reshape(height, width, 1)[:,p_[1]:,0][(k[:,p_[1]:,2] > k3) | (k[:,p_[1]:,2] < k4)]
    k[:p_[0],:,0][(k1 >= k[:p_[0],:,2]) & (k[:p_[0],:,2] >= k4)] = \
    (np.subtract(p_[0], k[:,:,0]) / p_[0]).reshape(height, width, 1)[:p_[0],:,0][(k1 >= k[:p_[0],:,2]) & (k[:p_[0],:,2] >= k4)]
    k[p_[0]:,:,0][(k3 >= k[p_[0]:,:,2]) & (k[p_[0]:,:,2] >= k2)] = \
    (np.subtract(k[:,:,0], p_[0]) / (height - p_[0])).reshape(height, width, 1)[p_[0]:,:,0][(k3 >= k[p_[0]:,:,2]) & (k[p_[0]:,:,2] >= k2)]

    cita = np.exp(-np.power(k[:,:,0] / thre,2))
    cita[minx:maxx,miny:maxy] = 1
    # 如果不需要局部变形，可以把cita的值全置为1
    cita[:,:]=1
    return cita

# 映射函数f(v)
def cal_fv(target_pts, W, A, Z, height, width, cita, img_coord):
    qstar = np.divide(np.matmul(W,target_pts), np.sum(W, axis=2).reshape(height,width,1))

    qhat = np.subtract(target_pts, qstar.reshape(height, width, 1, 2)).reshape(height, width, target_pts.shape[0], 1, 2)

    fv_ = np.sum(np.matmul(qhat, A),axis=2)

    fv = np.linalg.norm(Z[:,:,0,:,:],axis=2) / (np.linalg.norm(fv_,axis=3)+0.0000000001) * fv_[:,:,0,:] + qstar

    fv = (fv - img_coord) * cita.reshape(height, width, 1) + img_coord

    return fv

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
