"""Spatial wafer features with normalized coordinates, using only map pixels (0/1/2)."""
import numpy as np
from scipy import ndimage

CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']

def extract_features(a):
    a = np.asarray(a)
    valid = a > 0
    bad = a == 2
    yy, xx = np.indices(a.shape)
    x = (xx - (a.shape[1]-1)/2) / max((a.shape[1]-1)/2, 1)
    y = (yy - (a.shape[0]-1)/2) / max((a.shape[0]-1)/2, 1)
    r = np.sqrt(x*x+y*y)
    theta = np.arctan2(y,x)
    n = max(valid.sum(),1)
    b = max(bad.sum(),1)
    f = [bad.sum()/n]
    def rates(idx, bins):
        den = np.bincount(idx[valid],minlength=bins)
        num = np.bincount(idx[bad],minlength=bins)
        return num / np.maximum(den,1)
    radial = np.minimum((r*16).astype(int),15)
    f.extend(rates(radial,16))
    for k in [4,8,16]:
        ang = np.minimum(((theta+np.pi)/(2*np.pi)*k).astype(int),k-1)
        f.extend(np.sort(rates(ang,k)))
        for lo,hi in [(0,.5),(.5,.8),(.8,2)]:
            mask = valid & (r>=lo)&(r<hi)
            den = np.bincount(ang[mask],minlength=k)
            num = np.bincount(ang[mask&bad],minlength=k)
            f.extend(np.sort(num/np.maximum(den,1)))
    for k in [4,8]:
        gx=np.clip(((x+1)*k/2).astype(int),0,k-1)
        gy=np.clip(((y+1)*k/2).astype(int),0,k-1)
        # Sorting reduces some orientation sensitivity; full rotation invariance is not guaranteed.
        f.extend(np.sort(rates(gy*k+gx,k*k)))
    if bad.any():
        xb,yb,rb=x[bad],y[bad],r[bad]
        f.extend([xb.mean()**2+yb.mean()**2, xb.std(),yb.std()])
        f.extend(np.quantile(rb,[0,.1,.25,.5,.75,.9,1]))
        cov=np.cov(np.array([xb,yb])) if b>1 else np.zeros((2,2))
        ev=np.linalg.eigvalsh(cov)
        f.extend([*ev,ev[0]/max(ev[1],1e-8)])
    else:
        f.extend([0]*13)
    # Morphology at normalized resolution; nearest resampling can lose fine structures.
    ri=np.minimum((np.arange(64)*a.shape[0]/64).astype(int),a.shape[0]-1)
    ci=np.minimum((np.arange(64)*a.shape[1]/64).astype(int),a.shape[1]-1)
    z=bad[np.ix_(ri,ci)]
    for size in [1,2,3,5]:
        zz=z if size==1 else ndimage.binary_opening(z,structure=np.ones((size,size)))
        labs,count=ndimage.label(zz,np.ones((3,3)))
        sizes=np.sort(np.bincount(labs.ravel())[1:])[::-1]
        top=np.zeros(5); top[:min(5,len(sizes))]=sizes[:5]/4096
        f.extend([count/4096,zz.mean(),*top])
    for sigma in [1,2,4,8]:
        sm=ndimage.gaussian_filter(z.astype(float),sigma)
        f.extend(np.quantile(sm,[.5,.75,.9,.95,.99,1]))
    return np.nan_to_num(np.asarray(f,dtype=np.float32))
