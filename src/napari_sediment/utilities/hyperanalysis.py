import numpy as np
import pandas as pd
from .sediproc import spectral_clustering



def compute_vertical_correlations(image_mnfr):

    all_coef = []
    for i in range(image_mnfr.shape[2]):
        
        im = image_mnfr[1::,:,i]
        im_shift = image_mnfr[0:-1,:,i]
        
        all_coef.append(np.corrcoef(im.flatten(), im_shift.flatten())[0,1])
    all_coef = np.array(all_coef)

    return all_coef

def compute_end_members(pure, im_cube, im_cube_denoised, ppi_threshold, dbscan_eps):

    # recover pixel vectors from denoised image and actual image
    vects = im_cube_denoised[:, pure > ppi_threshold]
    vects_image = im_cube[:,pure > ppi_threshold] 
    
    # compute clustering
    labels = spectral_clustering(pixel_vectors=vects.T, dbscan_eps=dbscan_eps)

    end_members = []
    for ind in range(0, labels.max()+1):
        
        endmember = vects_image[:, labels==ind].mean(axis=1)
        end_members.append(endmember)

    end_members_raw = np.stack(end_members, axis=1)

    return end_members_raw, labels

def reduce_with_mnf(im_mnf, corr_coefficients, corr_threshold, max_index=None):
    """Reduce the number of bands of an image by keeping only mnf transformed bands
    with significant vertical correlation (noisy images have low band to band correlation)."""

    if max_index is not None:
        acceptable_range = np.arange(max_index)
    else:
        acceptable_range = np.arange(len(corr_coefficients))

    accepted_corr = corr_coefficients[acceptable_range]
    accepted_indices = acceptable_range[accepted_corr > corr_threshold]
    if len(accepted_indices) > 0:
        last_index = acceptable_range[accepted_corr > corr_threshold][-1]
    else:
        raise ValueError(f'No bands with correlation > {corr_threshold}')
    selected_bands = im_mnf[:,:, 0:last_index].copy()
    
    return selected_bands

def export_dim_reduction_data(export_path, eigenvals, all_coef, end_members,
                              bands_used):
    
    if eigenvals is not None:
        df = pd.DataFrame(eigenvals, columns=['eigenvalues'])
        df.to_csv(export_path.joinpath('eigenvalues.csv'), index=False)
    if all_coef is not None:
        df = pd.DataFrame(all_coef, columns=['correlation'])
        df.to_csv(export_path.joinpath('correlation.csv'), index=False)
    if end_members is not None:
        df = pd.DataFrame(end_members, columns=np.arange(end_members.shape[1]))
        df['bands'] = bands_used
        df.to_csv(export_path.joinpath('end_members.csv'), index=False)