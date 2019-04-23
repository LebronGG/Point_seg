import h5py
f = h5py.File('./data/indoor3d_ins_seg_hdf5/Area_1_hallway_7.h5')
data=f['data'][:]
label=f['label'][:]
print f.keys()