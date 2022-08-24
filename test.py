import torchimport osimport cv2from PIL import Imagefrom model.ESRGAN import ESRGAN, ESRGAN_tiny, ESRGAN_tiny_2Xfrom model.RFDN import RFDNfrom torchvision.utils import save_imageimport torchvision.transforms.functional as TFfrom collections import OrderedDictimport argparseYUV_flag = Truescale = 2parser = argparse.ArgumentParser()parser.add_argument('--gan_pth_path', default='./checkpoints_FishSR_RRDB_2X_2D/generator_49.pth')parser.add_argument('--psnr_pth_path', default='./checkpoints_FishSR_RRDB_2X_2D/generator_49.pth')parser.add_argument('--interp_pth_path', default='parameters/interp.pth')parser.add_argument('--lr_dir', default='F:/WJ_project/ESRGAN/eval/inputfishSR')parser.add_argument('--sr_dir', default='F:/WJ_project/ESRGAN/eval/outputfishSR')# parser.add_argument('--lr_dir', default='F:/WJ_project/ESRGAN/eval/blendtest')# parser.add_argument('--sr_dir', default='F:/WJ_project/ESRGAN/eval/blendtest_result')parser.add_argument('--alpha', type=float, default=0.5)args = parser.parse_args()net_PSNR = torch.load(args.psnr_pth_path, map_location='cpu')net_ESRGAN = torch.load(args.gan_pth_path, map_location='cpu')# net_PSNR = torch.load(args.psnr_pth_path)# net_ESRGAN = torch.load(args.gan_pth_path)net_interp = OrderedDict()for k, v_PSNR in net_PSNR.items():    v_ESRGAN = net_ESRGAN[k]    net_interp[k] = (1 - args.alpha) * v_PSNR + args.alpha * v_ESRGANif not os.path.exists(args.lr_dir):    # raise Exception('[!] No lr path')    print('No lr path')if not os.path.exists(args.sr_dir):    os.makedirs(args.sr_dir)device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# device = torch.device("cpu")def test_crop(input, crop=576, padding=0, scale=1):    _, channel, hhh, www,  = input.shape    hhh_s = scale * hhh    www_s = scale * www    crop_s = scale * crop    padding_s = scale * padding    stride = crop - padding    allh = hhh // stride + 1    allw = www // stride + 1    output = torch.zeros(1, channel, hhh_s, www_s).to(device)    count = torch.zeros(1, channel, hhh_s, www_s).to(device)    for idxh in range(allh):        for idxw in range(allw):            input_patch = input[:, :, idxh*crop-idxh*padding: (idxh+1)*crop-idxh*padding, idxw*crop-idxw*padding: (idxw+1)*crop-idxw*padding]            output[:, :, scale*idxh*crop-scale*idxh*padding: scale*(idxh+1)*crop-scale*idxh*padding, scale*idxw*crop-scale*idxw*padding: scale*(idxw+1)*crop-scale*idxw*padding] += net(input_patch)            count[:, :, scale*idxh*crop-scale*idxh*padding: scale*(idxh+1)*crop-scale*idxh*padding, scale*idxw*crop-scale*idxw*padding: scale*(idxw+1)*crop-scale*idxw*padding] += 1    output = output / count    return outputdef test_crop_blend(input, crop=576, padding=26, scale=1):    _, channel, hhh, www,  = input.shape    hhh_s = scale * hhh    www_s = scale * www    crop_s = scale * crop    padding_s = scale * padding    stride = crop - padding    allh = hhh // stride + 1    allw = www // stride + 1    current_patch = [[0] * allw for _ in range(allh)]    output = torch.zeros(1, channel, hhh_s, www_s).to(device)    count = torch.zeros(1, channel, hhh_s, www_s).to(device)    for idxh in range(allh):        for idxw in range(allw):            input_patch = input[:, :, idxh*crop-idxh*padding: (idxh+1)*crop-idxh*padding, idxw*crop-idxw*padding: (idxw+1)*crop-idxw*padding]            current_patch[idxh][idxw] = net(input_patch)    for idxh in range(allh):        for idxw in range(allw):            output[:, :, scale*idxh*crop-scale*idxh*padding: scale*(idxh+1)*crop-scale*idxh*padding, scale*idxw*crop-scale*idxw*padding: scale*(idxw+1)*crop-scale*idxw*padding] += current_patch[idxh][idxw]            count[:, :, scale * idxh * crop: scale * (idxh + 1) * crop, scale * idxw * crop: scale * (idxw + 1) * crop] += 1    for idxh in range(0, allh):        for idxw in range(0, allw):    # for idxh in range(0, 2):    #     for idxw in range(0, 2):            for padw in range(padding):                if idxh == 0:                    start = 0                    bias = 0                    end = crop * scale                elif idxh == allh - 1:                    bias = 0                    start = padding * scale-bias                    # end = padding * scale + 356#hhh_s - idxh * scale * crop                    end = hhh_s - (allh-1) * scale * (crop-padding)                else:                    start = padding * scale - 0                    end = crop * scale                aw = 1 * (padw) / padding                if idxw == allw-1:                    continue                temp = aw * current_patch[idxh][idxw][:, :, start:end, scale * crop - scale * (padding - padw):scale * crop - scale * (padding - padw) + scale]                output[:, :, start + scale * idxh * (crop - padding): scale * idxh * (crop - padding) + scale * crop -bias, scale * (idxw + 1) * (crop - padding) + scale * padw: scale * (idxw + 1) * (crop - padding) + scale * (padw + 1)] \                    -= temp                temp = (1 - aw) * current_patch[idxh][idxw + 1][:, :, start:end, scale * padw:scale * padw + scale]                output[:, :, start + scale * idxh * (crop - padding): scale * idxh * (crop - padding) + scale * crop-bias, scale * (idxw + 1) * (crop - padding) + scale * padw: scale * (idxw + 1) * (crop - padding) + scale * (padw + 1)] \                    -= temp            for padh in range(padding):                if idxw == allw - 1:                    # bottom = 412#www_s - idxw * scale * crop                    bottom = www_s - (allw-1) * scale * (crop-padding)                else:                    bottom = crop * scale                ah = 1 * (padh) / padding                if idxh == allh-1:                    continue                temp = ah * current_patch[idxh][idxw][:, :, scale * crop - scale * (padding - padh):scale * crop - scale * (padding - padh) + scale, :bottom]                output[:, :, scale * (idxh + 1) * (crop - padding) + scale * padh: scale * (idxh + 1) * (crop - padding) + scale * (padh + 1),scale * idxw * (crop - padding): scale * idxw * (crop - padding) + scale * crop] \                    -= temp                temp = (1 - ah) * current_patch[idxh + 1][idxw][:, :, scale * padh:scale * padh + scale, :bottom]                output[:, :, scale * (idxh + 1) * (crop - padding) + scale * padh: scale * (idxh + 1) * (crop - padding) + scale * (padh + 1),scale * idxw * (crop - padding): scale * idxw * (crop - padding) + scale * crop] \                    -= temp    output = output / count    return outputwith torch.no_grad():    # net = ESRGAN(3, 3, scale_factor=4)    # net = ESRGAN_tiny(1, 1, nf=32, gc=4, n_basic_block=5, scale_factor=2)    # net = ESRGAN_tiny(1, 1, nf=64, gc=16, n_basic_block=12, scale_factor=1)    net = ESRGAN_tiny_2X(1, 1, nf=32, gc=32, n_basic_block=12, scale_factor=2)    # net = RFDN(in_nc=1, nf=80, num_modules=4, out_nc=1, upscale=2)    # net = RFDN(in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4)    net.load_state_dict(net_interp)    net = net.to(device).eval()    # YUV_flag = False    for image_name in os.listdir(args.lr_dir):        input_RGB = cv2.imread(os.path.join(args.lr_dir, image_name))        if YUV_flag:            input_YUV = cv2.cvtColor(input_RGB, cv2.COLOR_BGR2YUV)            if scale == 2:                input_RGB_copy = cv2.resize(input_RGB, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)                input_YUV_copy = cv2.cvtColor(input_RGB_copy, cv2.COLOR_BGR2YUV)                input_U = input_YUV_copy[:, :, 1]                input_V = input_YUV_copy[:, :, 2]            else:                input_U = input_YUV[:, :, 1]                input_V = input_YUV[:, :, 2]            input_Y = input_YUV[:, :, 0]            input_Y_tensor = TF.to_tensor(Image.fromarray(input_Y)).to(device).unsqueeze(dim=0)            input_U_tensor = TF.to_tensor(Image.fromarray(input_U)).to(device).unsqueeze(dim=0)            input_V_tensor = TF.to_tensor(Image.fromarray(input_V)).to(device).unsqueeze(dim=0)        else:            input_Y_tensor = TF.to_tensor(Image.fromarray(input_RGB)).to(device).unsqueeze(dim=0)        import time        t0 = time.time()        # result_y = test_crop(input_Y_tensor, crop=288, padding=26, scale=scale)        result_y = test_crop_blend(input_Y_tensor, crop=576, padding=12, scale=scale)        result_y = torch.clamp(result_y, 0, 1)        t1 = time.time()        if YUV_flag:            # if scale == 2:            #     # result = result_y            #     result = torch.cat((result_y, result_u, result_v), 1)            # else:            result = torch.cat((result_y, input_U_tensor, input_V_tensor), 1)        else:            result = result_y        print(result.shape)        result = result[0,:,:,:]        result = result.cpu().numpy()        result = result.transpose(1, 2, 0)        result = (255 * result).astype("uint8")        print(result.shape)        if YUV_flag:            if scale == 2:                result = cv2.cvtColor(result, cv2.COLOR_YUV2BGR)            else:                result = cv2.cvtColor(result, cv2.COLOR_YUV2BGR)        cv2.imwrite(os.path.join(args.sr_dir, ''+image_name), result)        print('time:', t1-t0)        # result_RGB = TF.to_tensor(Image.fromarray(result_RGB)).to(device).unsqueeze(dim=0)        # print(result)        # save_image(result,  os.path.join(args.sr_dir, image_name))        print(f'save {image_name}')