import torch

mask=[]

mask_conv0 = torch.randn(64,3,3,3).cuda()/10
mask_conv1 = torch.randn(64,64,3,3).cuda()/10

mask_conv2 = torch.randn(128,64,3,3).cuda()/10
mask_conv3 = torch.randn(128,128,3,3).cuda()/10

mask_conv4 = torch.randn(256,128,3,3).cuda()/10
mask_conv5 = torch.randn(256,256,3,3).cuda()/10
mask_conv6 = torch.randn(256,256,3,3).cuda()/10

mask_conv7 = torch.randn(512,256,3,3).cuda()/10
mask_conv8 = torch.randn(512,512,3,3).cuda()/10
mask_conv9 = torch.randn(512,512,3,3).cuda()/10
mask_conv10 = torch.randn(512,512,3,3).cuda()/10
mask_conv11 = torch.randn(512,512,3,3).cuda()/10
mask_conv12 = torch.randn(512,512,3,3).cuda()/10
mask_fc0 = torch.randn(512,512).cuda()/10
mask_fc1 = torch.randn(512,512).cuda()/10

mask_fc2 = torch.randn(100,512).cuda()/10

mask.append(mask_conv0)
mask.append(mask_conv1)
mask.append(mask_conv2)
mask.append(mask_conv3)
mask.append(mask_conv4)
mask.append(mask_conv5)
mask.append(mask_conv6)
mask.append(mask_conv7)
mask.append(mask_conv8)
mask.append(mask_conv9)
mask.append(mask_conv10)
mask.append(mask_conv11)
mask.append(mask_conv12)
mask.append(mask_fc0)
mask.append(mask_fc1)
mask.append(mask_fc2)

torch.save(mask, 'mask_rand2.dat')
'''
mask_conv0_weight = torch.randn(64,3,3,3).cuda()/10
mask_conv0_bias = torch.randn(64).cuda()/10
mask_conv1_weight = torch.randn(64,64,3,3).cuda()/10
mask_conv1_bias = torch.randn(64).cuda()/10

mask_conv2_weight = torch.randn(128,64,3,3).cuda()/10
mask_conv2_bias = torch.randn(128).cuda()/10
mask_conv3_weight = torch.randn(128,128,3,3).cuda()/10
mask_conv3_bias = torch.randn(128).cuda()/10

mask_conv4_weight = torch.randn(256,128,3,3).cuda()/10
mask_conv4_bias = torch.randn(256).cuda()/10
mask_conv5_weight = torch.randn(256,256,3,3).cuda()/10
mask_conv5_bias = torch.randn(256).cuda()/10
mask_conv6_weight = torch.randn(256,256,3,3).cuda()/10
mask_conv6_bias = torch.randn(256).cuda()/10

mask_conv7_weight = torch.randn(512,256,3,3).cuda()/10
mask_conv7_bias = torch.randn(512).cuda()/10
mask_conv8_weight = torch.randn(512,512,3,3).cuda()/10
mask_conv8_bias = torch.randn(512).cuda()/10
mask_conv9_weight = torch.randn(512,512,3,3).cuda()/10
mask_conv9_bias = torch.randn(512).cuda()/10
mask_conv10_weight = torch.randn(512,512,3,3).cuda()/10
mask_conv10_bias = torch.randn(512).cuda()/10
mask_conv11_weight = torch.randn(512,512,3,3).cuda()/10
mask_conv11_bias = torch.randn(512).cuda()/10
mask_conv12_weight = torch.randn(512,512,3,3).cuda()/10
mask_conv12_bias = torch.randn(512).cuda()/10
mask_fc0_weight = torch.randn(512,512).cuda()/10
mask_fc0_bias = torch.randn(512).cuda()/10
mask_fc1_weight = torch.randn(512,512).cuda()/10
mask_fc1_bias = torch.randn(512).cuda()/10

mask_fc2_weight = torch.randn(100,512).cuda()/10
mask_fc2_bias = torch.randn(100).cuda()/10

mask.append(mask_conv0_weight)
mask.append(mask_conv0_bias)
mask.append(mask_conv1_weight)
mask.append(mask_conv1_bias)
mask.append(mask_conv2_weight)
mask.append(mask_conv2_bias)
mask.append(mask_conv3_weight)
mask.append(mask_conv3_bias)
mask.append(mask_conv4_weight)
mask.append(mask_conv4_bias)
mask.append(mask_conv5_weight)
mask.append(mask_conv5_bias)
mask.append(mask_conv6_weight)
mask.append(mask_conv6_bias)
mask.append(mask_conv7_weight)
mask.append(mask_conv7_bias)
mask.append(mask_conv8_weight)
mask.append(mask_conv8_bias)
mask.append(mask_conv9_weight)
mask.append(mask_conv9_bias)
mask.append(mask_conv10_weight)
mask.append(mask_conv10_bias)
mask.append(mask_conv11_weight)
mask.append(mask_conv11_bias)
mask.append(mask_conv12_weight)
mask.append(mask_conv12_bias)
mask.append(mask_fc0_weight)
mask.append(mask_fc0_bias)
mask.append(mask_fc1_weight)
mask.append(mask_fc1_bias)
mask.append(mask_fc2_weight)
mask.append(mask_fc2_bias)

torch.save(mask, 'mask_wb_rand.dat')
'''
