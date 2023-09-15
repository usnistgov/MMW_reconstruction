#
#
# This version is current as of 2023-09-15
#
# This code was mostly written by Praful Gupta
# in collaboration with NIST during his PhD at UT Austin.
# It is now maintained by Jack Glover of NIST
#
# 2023-01-30 Recon take ~140 seconds per view on my laptop with 60 % bandwidth
# 2023-01-30 Recon take ~70 seconds per view on my laptop with 10 % bandwidth
# 2022-06-12 Works with Python 3.10 but not 3.11 (because of lack of libraries)
# 2022-06-12 Runs in 1 to 2 minutes on my laptop, depending on reconstruction settings
#

#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import time
import matplotlib.pyplot as plt
import tsahelper as tsa # this is just a library to read in ahi files
from interpolation.splines import eval_linear
from interpolation.splines import extrap_options as xto
# from scipy.interpolate import interp2d
from scipy import ndimage
import os
from PIL import Image



def ensure_dir(name):
    if not os.path.isdir(name):
        os.makedirs(name)

def export_figure_matplotlib(arr, f_name, dpi=200, resize_fact=1, plt_show=False):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1]/dpi, arr.shape[0]/dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, cmap = 'gray')
    plt.savefig(f_name, dpi=(dpi * resize_fact))
    if plt_show:
        plt.show()
    else:
        plt.close()



def reconstruct_from_ahi(ahi_filenames,recon_freq,hamming=True,angles=[0]):
    eps = 1e-6
    start = time.time()

    z_offset_by2 = 75

    theta_starts = np.append(np.arange(np.pi/2, np.pi/2 + 3*np.pi/8 + eps, np.pi/8), np.arange(-np.pi, -np.pi + 11*np.pi/8 + eps, np.pi/8))

    for ahi_filename in ahi_filenames:
        print(ahi_filename)
        base_filename = ahi_filename[:-4]
        h = tsa.read_header(ahi_filename)


        # These are all the experimental params we read from the file
        Nf         = int(h['num_t_pts'][0])
        step       = int(h['band_width'])/Nf
        f_mid      = int(h['frequency'])
        f_arr_full = (f_mid + np.arange(-Nf/2, Nf/2)*step ) *1e6 # in Hz -- from 10 GHz to 40 GHz
        R0          = h['scanner_radius'] #  0.63
        dz         = h['y_inc'][0]  # 0.003175 = 2.0955/660
        Ntheta_input     = int(h['num_x_pts'])
        theta_step  = h['theta_step']

        #print('theta_step (rad)',theta_step)
        #print('theta_step_rad', theta_step*180/np.pi)

        Nz = 660
        z_scan = np.arange(0, Nz)*dz
        #### dimension of x a:qnd y in spatial domain
        Nx = 512
        Ny = 512
        c = 299792458.0

        for recon_cf in recon_freq.keys():
            print(' recon_cf',recon_cf)
            for recon_bw_perc in recon_freq[recon_cf]:
                print('  recon_bw_perc',recon_bw_perc)

                output_dir = os.path.join('./output_images')
                ensure_dir(output_dir)

                for ax_ind, theta_start in enumerate(theta_starts):
                    if not(ax_ind in angles): continue
                    ### theta_start
                    f_name = f'{output_dir}/{base_filename}_cf_{recon_cf}_bw_{recon_bw_perc}_ang_{ax_ind}.png'
                    if os.path.isfile(f_name): continue
                    print('  ax_ind', ax_ind)
                    print('  ',int(time.time() - start), 'seconds elapsed')

                    s_full = tsa.read_data(ahi_filename)
                    s_full = s_full.swapaxes(0,2)

                    theta_arr  = theta_start + theta_step * np.linspace(-Ntheta_input/2,+Ntheta_input/2,num=Ntheta_input)*1.0 # Changing theta array at each iteration

                    ### s, theta array and freq array

                    th0 = 0
                    half_range = np.pi/4
                    ind        = np.where(np.logical_and(theta_arr>=th0-half_range , theta_arr<=th0+half_range))[0]    # cut off redundant info
                    theta_diff = theta_arr[1]-theta_arr[0]
                    theta_arr = np.arange(-len(ind),len(ind))*theta_diff
                    s_full     = s_full[:,ind, :]

                    ## pad window, s_full
                    window = np.pad(np.hamming(int(len(ind))), (int(len(theta_arr)/4), int(len(theta_arr)) - int(len(ind)) - int(len(theta_arr)/4)), 'constant', constant_values=0)
                    s_full = np.pad(s_full, ((0,0), (int(len(theta_arr)/4), int(len(theta_arr)) - int(len(ind)) - int(len(theta_arr)/4)), (0,0)), 'constant', constant_values=0)

                    # print(len(theta_arr),' angles ranging from ',theta_arr[0]*180/3.14,' to ',theta_arr[-1]*180/3.14,' degrees')

                    f_min = recon_cf*1e9 - 0.01*recon_bw_perc*recon_cf*1e9
                    f_max = recon_cf*1e9 + 0.01*recon_bw_perc*recon_cf*1e9

                    # print(f_min, f_max)
                    ind_f  = np.where(np.logical_and(f_arr_full>f_min, f_arr_full<f_max))[0]
                    s = s_full[ind_f,:,:].astype(np.complex64)
                    s_full = None
                    f_arr    = f_arr_full[ind_f]

                    s = np.pad(s, ((0,), (0,), (z_offset_by2,)), 'constant', constant_values=0)
                    window = window[np.newaxis, :, np.newaxis]
                    window = np.tile(window, (s.shape[0], 1, s.shape[2]))
                    s = s*window

                    if hamming:
                        f_window = np.hamming(s.shape[0])
                        s = s*f_window[:,None,None]

                    freq = f_arr
                    theta = theta_arr

                    f_arr = None
                    theta_arr = None

                    Nf = len(freq)
                    Ntheta = len(theta)
                    Nz = s.shape[2]

                    ## angular wave number
                    kf = 2*np.pi*freq/c

                    ## z sweep

                    # kz=np.linspace(-np.pi/dz, np.pi/dz-2*np.pi/dz/(Nz-1),Nz)
                    kz = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(Nz, d=dz))

                    ind_f, ind_theta, ind_z = np.indices((int(Nf), Ntheta, Nz))

                    ## computing Kr
                    kr_sq = 4*kf[ind_f]**2 - kz[ind_z]**2
                    kr = np.sqrt(kr_sq*(kr_sq>0))
                    kr_sq = None

                    ##H2 is similar to the phase shift term in Sheen's paper
                    H2= R0*np.exp(1j*R0*kr*np.cos(theta[ind_theta]))*window
                    window = None
                    kz_shift = np.exp(1j* kz[ind_z] * abs(z_scan[int(Nz/2)-1]))#*np.exp(-2j*kf[ind_f]*R0)
                    s  = np.fft.fftshift(np.fft.fft(s, axis = 2), axes = [2])*kz_shift ## Shiyong's paper does this shifting in kz direction..

                    s  = np.fft.fft(s, Ntheta, axis = 1)
                    H2 = np.fft.fft(H2, Ntheta, axis = 1)
                    sH = s*H2

                    s  = None
                    sH = np.fft.ifftshift(np.fft.ifft(sH, Ntheta, axis = 1), axes = [1])

                    ## spatial frequency domain
                    # kx=np.linspace(-np.max(kr),np.max(kr), Nx)
                    # ky=np.linspace(-np.max(kr),np.max(kr), Ny)
                    L_x = 1.0
                    kx = np.linspace(-0.5,0.5, Nx) * 2*np.pi*Nx/L_x
                    ky = kx*1.0

                    #### interpolation from kr, theta to kx, ky for each kz;
                    ####
                    theta_grid_in = theta[ind_theta]

                    ind_x, ind_y, ind_z = np.indices((Nx, Ny, Nz))
                    kr_grid_out = np.sqrt(kx[ind_x]**2 + ky[ind_y]**2)
                    theta_grid_out = np.arctan2(ky[ind_y], kx[ind_x])

                    ssH = np.zeros((Nx, Ny, Nz), dtype=np.complex64)

                    # ## check if this can be optimized more
                    for ind_kz in range(Nz):
                        #if ind_kz%50==0: print(ind_kz)
                        out_grid = np.vstack((kr_grid_out[:,:,ind_kz].ravel(), theta_grid_out[:,:,ind_kz].ravel())).T
                        # print(kr.shape, theta_grid_in.shape)
                        # aa
                        grid = (kr[:,0,ind_kz],  theta_grid_in[0,:,ind_kz])
                        temp_real = eval_linear(grid, np.real(sH[:,:,ind_kz]), out_grid, xto.CONSTANT) # This works fine
                        temp_imag = eval_linear(grid, np.imag(sH[:,:,ind_kz]), out_grid, xto.CONSTANT) # This works fine
                        temp = temp_real + 1j*temp_imag
                        ssH[:,:,ind_kz] = temp.reshape((Nx, Ny))

                    ## ifft to get the final image in spatial domain
                    sH = None
                    image_3d = np.fft.ifftshift(np.fft.ifftn(ssH, [Nx, Ny, Nz]))
                    ssH = None

                    #### normalizing image
                    factor   = 1.0/(0.8*np.max(np.abs(image_3d)))
                    #factor   = 168
                    image_3d = np.abs(image_3d)*factor
                    image_3d[image_3d>1]=1

                    #print('finished recon')

                    # print('time taken %d secs'%(time.time()-start))

                    # create front max val projection
                    print(image_3d.shape)

                    mid0 = int(image_3d.shape[0]/2)
                    mid1 = int(image_3d.shape[1] / 2)
                    w = 15
                    #image_3d[mid0-w:mid0+w,mid1-w:mid1+w,:] = 0

                    arr =ndimage.rotate(np.max(image_3d,axis=0), 90)
                    arr = arr[z_offset_by2:-z_offset_by2, :]
                    export_figure_matplotlib(arr, f_name, dpi=200, resize_fact=1, plt_show=False)

                    # create top max val projection
                    if False:
                        arr_top =ndimage.rotate((image_3d[10:-10,10:-1,139+75]), 90)
                        arr_top = ndimage.rotate((image_3d[10:-10, 10:-1, 139 + 75 +125]), 90)
                        arr_top = ndimage.rotate(np.max(image_3d, axis=2), 90)
                        export_figure_matplotlib(arr_top, f_name+'_top.png', dpi=200, resize_fact=1, plt_show=False)




if __name__=="__main__":
    # recon_freq is a dict that specifies the reconstruction parameters
    # the key gives the central frequency in GHz
    # the value gives an array of bandwidth values, expressed as a percentage
    # e.g. 25:[60] will run from 25*(1-0.6)= 10 GHz to 25*(1+0.6)= 40 GHz

    recon_freq = { 25:[60,10], 35:[10], 15:[10], }

    angles = [0]

    ahi_filenames = ['example.ahi']

    reconstruct_from_ahi(ahi_filenames, recon_freq, angles=angles ) # np.arange(20) )

    #compare reconstruction to aps
    angle = 0
    recon_filename = f'output_images/example_cf_25_bw_60_ang_{angle}.png'
    img_recon = np.asarray(Image.open(recon_filename))[:,:,0]

    plt.subplot(1,3,1)
    plt.title('recon from .ahi')
    plt.imshow(img_recon)

    aps_imgs = tsa.read_data('example.aps')
    aps_img = aps_imgs[:,:,0]
    aps_img = ndimage.rotate(aps_img, 90)

    plt.subplot(1,3,2)
    plt.title('.aps')
    plt.imshow(aps_img)

    diff = (img_recon*1.0/np.mean(img_recon) - aps_img*1.0/np.mean(aps_img))/(aps_img*1.0/np.mean(aps_img))
    plt.subplot(1,3,3)
    plt.title('diff %')
    plt.imshow(diff)

    big_diff = diff[img_recon > 5]
    print('median err on target',np.median(np.abs(big_diff.ravel())))

    plt.show()


