# FishSR
流程说明
第一步：使用data_tool内Fisheye_simu工具，从高清图仿真出模仿鱼眼矫正的退化图，高清图建议下载开源的DIV2K等数据集合 第二步：使用data_tool内coner_crop工具，同时切分高低清图四角区域形成配对 第三步：开启训练，如上述说明，使用FishSR_project内的main.py脚本，详细内容请参阅论文

推理说明

cd codes

#for RRDB model inference:
python run.py --which_model RRDB --corner_area_pth ../SR_model/RRDB_corner_area.pth --mid_area_pth ../SR_model/RRDB_mid_area.pth --matrix_pth ../matrix --lr_dir ../input --sr_dir ../output --alpha 0.8 --scale 2 --YUV_flag True --crop_size 576 --padding 10 --img_distort_width 1920 --img_distort_height 1080 --gamma 1.4 --bilateral_kernel 0 --save_mid_result True

#for RFDN model(light-weight) inference:
python run.py --which_model RFDN --corner_area_pth ../SR_model/RFDN_corner_area.pth --mid_area_pth ../SR_model/RFDN_mid_area.pth --matrix_pth ../matrix --lr_dir ../input --sr_dir ../output --alpha 0.8 --scale 2 --YUV_flag True --crop_size 576 --padding 10 --img_distort_width 1920 --img_distort_height 1080 --gamma 1.0 --bilateral_kernel 0 --save_mid_result True

#for low-distorted inputs, set parameter 'alpha' lower than 0.5
