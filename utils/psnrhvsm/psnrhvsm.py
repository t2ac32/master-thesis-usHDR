# Generated with SMOP  0.41
from libsmop import *
# psnrhvsm.m

    
@function
def psnrhvsm(img1=None,img2=None,wstep=None,*args,**kwargs):
    varargin = psnrhvsm.varargin
    nargin = psnrhvsm.nargin

    #========================================================================
    
    # Calculation of PSNR-HVS-M and PSNR-HVS image quality measures
    
    # PSNR-HVS-M is Peak Signal to Noise Ratio taking into account 
# Contrast Sensitivity Function (CSF) and between-coefficient   
# contrast masking of DCT basis functions
# PSNR-HVS is Peak Signal to Noise Ratio taking into account only CSF
    
    # Copyright(c) 2006 Nikolay Ponomarenko 
# All Rights Reserved
    
    # Homepage: http://ponomarenko.info, E-mail: nikolay{}ponomarenko.info
    
    #----------------------------------------------------------------------
    
    # Permission to use, copy, or modify this software and its documentation
# for educational and research purposes only and without fee is hereby
# granted, provided that this copyright notice and the original authors'
# names appear on all copies and supporting documentation. This program
# shall not be used, rewritten, or adapted as the basis of a commercial
# software or hardware product without first obtaining permission of the
# authors. The authors make no representations about the suitability of
# this software for any purpose. It is provided "as is" without express
# or implied warranty.
    
    #----------------------------------------------------------------------
    
    # This is an implementation of the algorithm for calculating the PSNR-HVS-M
# or PSNR-HVS between two images. Please refer to the following papers:
    
    # PSNR-HVS-M:
# [1] Nikolay Ponomarenko, Flavia Silvestri, Karen Egiazarian, Marco Carli, 
#     Jaakko Astola, Vladimir Lukin, "On between-coefficient contrast masking 
#     of DCT basis functions", CD-ROM Proceedings of the Third International 
#     Workshop on Video Processing and Quality Metrics for Consumer Electronics 
#     VPQM-07, Scottsdale, Arizona, USA, 25-26 January, 2007, 4 p.
    
    # PSNR-HVS:
# [2] K. Egiazarian, J. Astola, N. Ponomarenko, V. Lukin, F. Battisti, 
#     M. Carli, New full-reference quality metrics based on HVS, CD-ROM 
#     Proceedings of the Second International Workshop on Video Processing 
#     and Quality Metrics, Scottsdale, USA, 2006, 4 p.
    
    # Kindly report any suggestions or corrections to uagames{}mail.ru
    
    #----------------------------------------------------------------------
    
    # Input : (1) img1: the first image being compared
#         (2) img2: the second image being compared
#         (3) wstep: step of 8x8 window to calculate DCT 
#             coefficients. Default value is 8.
    
    # Output: (1) p_hvs_m: the PSNR-HVS-M value between 2 images.
#             If one of the images being compared is regarded as 
#             perfect quality, then PSNR-HVS-M can be considered as the
#             quality measure of the other image.
#             If compared images are visually undistingwished, 
#             then PSNR-HVS-M = 100000.
#         (2) p_hvs: the PSNR-HVS value between 2 images.
    
    # Default Usage:
#   Given 2 test images img1 and img2, whose dynamic range is 0-255
    
    #   [p_hvs_m, p_hvs] = psnrhvsm(img1, img2);
    
    # See the results:
    
    #   p_hvs_m  # Gives the PSNR-HVS-M value
#   p_hvs    # Gives the PSNR-HVS value
    
    #========================================================================
    
    if nargin < 2:
        p_hvs_m=- Inf
# psnrhvsm.m:77
        p_hvs=- Inf
# psnrhvsm.m:78
        return p_hvs_m,p_hvs
    
    if size(img1) != size(img2):
        p_hvs_m=- Inf
# psnrhvsm.m:83
        p_hvs=- Inf
# psnrhvsm.m:84
        return p_hvs_m,p_hvs
    
    if nargin > 2:
        step=copy(wstep)
# psnrhvsm.m:89
    else:
        step=8
# psnrhvsm.m:91
    
    img1=double(img1)
# psnrhvsm.m:94
    img2=double(img2)
# psnrhvsm.m:95
    LenXY=size(img1)
# psnrhvsm.m:97
    LenY=LenXY(1)
# psnrhvsm.m:97
    LenX=LenXY(2)
# psnrhvsm.m:97
    CSFCof=concat([[1.608443,2.339554,2.573509,1.608443,1.072295,0.643377,0.50461,0.421887],[2.144591,2.144591,1.838221,1.354478,0.989811,0.443708,0.428918,0.467911],[1.838221,1.979622,1.608443,1.072295,0.643377,0.451493,0.372972,0.459555],[1.838221,1.513829,1.169777,0.887417,0.50461,0.295806,0.321689,0.415082],[1.429727,1.169777,0.695543,0.459555,0.378457,0.236102,0.249855,0.334222],[1.072295,0.735288,0.467911,0.402111,0.317717,0.247453,0.227744,0.279729],[0.525206,0.402111,0.329937,0.295806,0.249855,0.212687,0.214459,0.254803],[0.357432,0.279729,0.270896,0.262603,0.229778,0.257351,0.249855,0.25995]])
# psnrhvsm.m:99
    # see an explanation in [2]
    
    MaskCof=concat([[0.390625,0.826446,1.0,0.390625,0.173611,0.0625,0.038447,0.026874],[0.694444,0.694444,0.510204,0.277008,0.147929,0.029727,0.027778,0.033058],[0.510204,0.591716,0.390625,0.173611,0.0625,0.030779,0.021004,0.031888],[0.510204,0.346021,0.206612,0.118906,0.038447,0.013212,0.015625,0.026015],[0.308642,0.206612,0.073046,0.031888,0.021626,0.008417,0.009426,0.016866],[0.173611,0.081633,0.033058,0.024414,0.015242,0.009246,0.007831,0.011815],[0.041649,0.024414,0.016437,0.013212,0.009426,0.00683,0.006944,0.009803],[0.01929,0.011815,0.01108,0.010412,0.007972,0.01,0.009426,0.010203]])
# psnrhvsm.m:109
    # see an explanation in [1]
    
    S1=0
# psnrhvsm.m:119
    S2=0
# psnrhvsm.m:119
    Num=0
# psnrhvsm.m:119
    X=1
# psnrhvsm.m:120
    Y=1
# psnrhvsm.m:120
    while Y <= LenY - 7:

        while X <= LenX - 7:

            A=img1(arange(Y,Y + 7),arange(X,X + 7))
# psnrhvsm.m:123
            B=img2(arange(Y,Y + 7),arange(X,X + 7))
# psnrhvsm.m:124
            A_dct=dct2(A)
# psnrhvsm.m:125
            B_dct=dct2(B)
# psnrhvsm.m:125
            MaskA=maskeff(A,A_dct)
# psnrhvsm.m:126
            MaskB=maskeff(B,B_dct)
# psnrhvsm.m:127
            if MaskB > MaskA:
                MaskA=copy(MaskB)
# psnrhvsm.m:129
            X=X + step
# psnrhvsm.m:131
            for k in arange(1,8).reshape(-1):
                for l in arange(1,8).reshape(-1):
                    u=abs(A_dct(k,l) - B_dct(k,l))
# psnrhvsm.m:134
                    S2=S2 + (dot(u,CSFCof(k,l))) ** 2
# psnrhvsm.m:135
                    if logical_or((k != 1),(l != 1)):
                        if u < MaskA / MaskCof(k,l):
                            u=0
# psnrhvsm.m:138
                        else:
                            u=u - MaskA / MaskCof(k,l)
# psnrhvsm.m:140
                    S1=S1 + (dot(u,CSFCof(k,l))) ** 2
# psnrhvsm.m:143
                    Num=Num + 1
# psnrhvsm.m:144

        X=1
# psnrhvsm.m:148
        Y=Y + step
# psnrhvsm.m:148

    
    if Num != 0:
        S1=S1 / Num
# psnrhvsm.m:152
        S2=S2 / Num
# psnrhvsm.m:152
        if S1 == 0:
            p_hvs_m=100000
# psnrhvsm.m:154
        else:
            p_hvs_m=dot(10,log10(dot(255,255) / S1))
# psnrhvsm.m:156
        if S2 == 0:
            p_hvs=100000
# psnrhvsm.m:159
        else:
            p_hvs=dot(10,log10(dot(255,255) / S2))
# psnrhvsm.m:161
    
    
@function
def maskeff(z=None,zdct=None,*args,**kwargs):
    varargin = maskeff.varargin
    nargin = maskeff.nargin

    # Calculation of Enorm value (see [1])
    m=0
# psnrhvsm.m:167
    MaskCof=concat([[0.390625,0.826446,1.0,0.390625,0.173611,0.0625,0.038447,0.026874],[0.694444,0.694444,0.510204,0.277008,0.147929,0.029727,0.027778,0.033058],[0.510204,0.591716,0.390625,0.173611,0.0625,0.030779,0.021004,0.031888],[0.510204,0.346021,0.206612,0.118906,0.038447,0.013212,0.015625,0.026015],[0.308642,0.206612,0.073046,0.031888,0.021626,0.008417,0.009426,0.016866],[0.173611,0.081633,0.033058,0.024414,0.015242,0.009246,0.007831,0.011815],[0.041649,0.024414,0.016437,0.013212,0.009426,0.00683,0.006944,0.009803],[0.01929,0.011815,0.01108,0.010412,0.007972,0.01,0.009426,0.010203]])
# psnrhvsm.m:169
    # see an explanation in [1]
    
    for k in arange(1,8).reshape(-1):
        for l in arange(1,8).reshape(-1):
            if logical_or((k != 1),(l != 1)):
                m=m + dot((zdct(k,l) ** 2),MaskCof(k,l))
# psnrhvsm.m:182
    
    pop=vari(z)
# psnrhvsm.m:186
    if pop != 0:
        pop=(vari(z(arange(1,4),arange(1,4))) + vari(z(arange(1,4),arange(5,8))) + vari(z(arange(5,8),arange(5,8))) + vari(z(arange(5,8),arange(1,4)))) / pop
# psnrhvsm.m:188
    
    m=sqrt(dot(m,pop)) / 32
# psnrhvsm.m:190
    
    
@function
def vari(AA=None,*args,**kwargs):
    varargin = vari.varargin
    nargin = vari.nargin

    d=dot(var(ravel(AA)),length(ravel(AA)))
# psnrhvsm.m:193