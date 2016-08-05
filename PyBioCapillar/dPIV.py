import openpiv.tools
import openpiv.process
import openpiv.scaling
import tifffile
import numpy as np
import matplotlib.pyplot as pp
import scipy.signal as ss
import os
import sys
os.chdir(sys.argv[1])
FileFlow=sys.argv[2]
FoldOut='FlowField'
DeltaT=1##h
Data=tifffile.imread(FileFlow)
Data1=Data.astype(np.int32)
Ws=10
Densty=5#the larger the less dense arroy field, smaller than Ws
PercFilt=50
conv_l=5
#pp.ion()
#pp.show()
#pp.clf()
if os.path.exists(FoldOut)==0:
    os.mkdir(FoldOut)
Tpoints=Data.shape[0]-1
X_Mat=np.zeros(Tpoints)
for kat in np.arange(Tpoints):
    frame_a=Data1[kat,:,:]
    frame_b=Data1[kat+1,:,:]
    u, v, sig2noise = openpiv.process.extended_search_area_piv( frame_a, frame_b, window_size=Ws, overlap=Ws-Densty, dt=DeltaT, search_area_size=Ws, sig2noise_method='peak2peak' )
    f_aMean=ss.convolve2d(frame_a,np.ones([conv_l*2,conv_l*2]),mode='same')
    f_aMean=f_aMean/((conv_l*2)**2)
    f_aStd=ss.convolve2d((frame_a-f_aMean)**2,np.ones([conv_l*2,conv_l*2]),mode='same')
    f_aStd=(f_aStd/((conv_l*2)**2))**0.5
    v=ss.convolve2d(v,np.ones([conv_l,conv_l]),mode='same')
    u=ss.convolve2d(u,np.ones([conv_l,conv_l]),mode='same')
    vv=np.hypot(v,u)
    ValidPos=np.zeros(vv.shape)+np.nan
    vv_1d=vv.ravel()
    vv_1dnan=vv_1d[~np.isnan(vv_1d)]
    Thres=np.percentile(vv_1dnan,PercFilt)
    #len(vv_1d[vv_1d<Thres])
    ValidPos[vv<Thres]=vv[vv<Thres]
    ValidPos[~np.isnan(ValidPos)]=ValidPos[~np.isnan(ValidPos)]    
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=Ws, overlap=Ws-Densty )
    #u, v, mask2 = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.2 )
    #vals=sig2noise.ravel()
    posts = np.vstack([x.ravel(), y.ravel()])
    postsInt=posts.astype(np.int)
    #P50=np.percentile(f_a.ravel(),50)
    #for katp in np.arange(posts.shape[1]):
    #    if f_a[postsInt[0,katp],postsInt[1,katp]]<P50:
    #        ValidPos[]=0
            
    mask=ValidPos>0
    pp.matshow(Data[kat,:,:])
    flipYMat=np.ones(y.shape)*Data1.shape[1]-1
    yFlipped=flipYMat-y
    x_0=np.zeros(x.shape)
    x_0[mask]=x[mask]
    yFlipped_0=np.zeros(yFlipped.shape)
    yFlipped_0[mask]=yFlipped[mask]
    v_0=np.zeros(v.shape)
    v_0[mask]=v[mask]
    u_0=np.zeros(u.shape)
    u_0[mask]=u[mask]
    pp.quiver(x_0,yFlipped_0,u_0,v_0,alpha=0.5)##pp.quiver(x[mask],yFlipped[mask],u[mask],v[mask])##pp.quiver(x,yFlipped,u,v)##pp.quiver(x,y,u,v)
    pp.savefig(FoldOut+'/TempTest_'+str(kat)+'.png',dpi=120)
    pp.close()
    DataSave=np.zeros([x_0.shape[0],x_0.shape[1],4])
    DataSave[:,:,0]=x_0
    DataSave[:,:,1]=yFlipped_0
    DataSave[:,:,2]=u_0
    DataSave[:,:,3]=v_0
    np.save(FoldOut+'/TempTest_'+str(kat)+'.npy',DataSave,allow_pickle=False, fix_imports=False)

    
    

    #raw_input('Press <ENTER> to continue')



##u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )


