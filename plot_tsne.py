import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

main_dir = '/home/Daejeon/'
cancer = 'tcga_stad'
folder_fn = 'stad_dinov2_sdkd_full_fold4_xdeepPrompt_molecular_L0L1feat_ep7999'
feature_folder_ours = f'{main_dir}/jingwei/{cancer}/{folder_fn}/'
feature_folder_gigapath = f'{main_dir}/dataset/features/{cancer}/prov_gigapath/s1_patchToken_regional_L0L1/' 
feature_folder_uni = f"{main_dir}/jingwei/{cancer}/UNI_feat_3scale_agg_patfeat/"
feature_folder_pathoduet = f"{main_dir}/jingwei/{cancer}/pathoduet_patfeat/"
feature_folder_gpfm = f"{main_dir}/jingwei/{cancer}/GPFM_feat/"
task = 'molecular'
fold = folder_fn.split('fold')[1].split('_')[0]
fold_dir = f'{main_dir}/jingwei/{cancer}/{cancer}_{task}.csv'


label = pd.read_csv(fold_dir)

tr_te_split =  label.loc[:, ['P_ID', f'fold_{fold}', f'{task}']]
tr_te_split.columns =['folder_name', 'split', 'label']
val_patient = list(tr_te_split[tr_te_split['split']=='val'])#['folder_name'].to_numpy())
train_patient = list(tr_te_split[tr_te_split['split']=='train'])#['folder_name'].to_numpy())
test_patient = tr_te_split[tr_te_split['split']=='test']#['folder_name'].to_numpy())
te_0_list = test_patient[test_patient['label']==0]
te_1_list = test_patient[test_patient['label']==1]

save_dir = f'/home/Alexandrite/jingwei/tsne/kd/{folder_fn}_test/'
os.makedirs(save_dir, exist_ok=True)
slide_num = 10

len_0 = len(te_0_list)
len_1 = len(te_1_list)
if len_0 < slide_num or len_1 < slide_num:
    slide_num = min(len_0, len_1)

print("Updated slide_num:", slide_num)

for ii in range (0, 50):
    #try:
        #ii = ii + 30
        print ('plotting ', ii)
te_0_slide = te_0_list.sample(n=slide_num, replace=True)['folder_name'].to_numpy()
te_1_slide = te_1_list.sample(n=slide_num, replace=True)['folder_name'].to_numpy()
te_slide = np.concatenate([te_0_slide, te_1_slide])
te_label = np.concatenate([[0]*slide_num, [1]*slide_num])
sli_list, gt_label = te_slide, te_label 
slide_list, all_feat_ours, all_feat_uni, all_feat_giga, all_feat_pathoduet, all_feat_gpfm = [], [], [], [], [], []
all_label, all_label_our, all_label_uni, all_label_giga, all_label_pathoduet, all_label_gpfm =  [], [], [], [], [], []
for ind in range (len(sli_list)):
    try:
        slide = sli_list[ind]
        gt = gt_label[ind]
        #our_feat = np.load(glob.glob(f"{feature_folder_ours}/{slide}*")[0])
        giga_feat = np.load(glob.glob(f"{feature_folder_gigapath}/{slide}*")[0])#.reshape(-1, 1536)
        #pathoduet_feat = np.load(glob.glob(f"{feature_folder_pathoduet}/{slide}*")[0])
        #gpfm_feat = np.load(glob.glob(f"{feature_folder_gpfm}/{slide}*")[0])
        #ni_feat = np.load(glob.glob(f"{feature_folder_uni}/{slide}*")[0]).reshape(-1, 1024)#[:,-256:,0,:]
        #if len(our_feat) != 0 and len(giga_feat) != 0 and len(pathoduet_feat) != 0 and len(gpfm_feat) != 0 and len(uni_feat) != 0:
        #print (f" {slide} ours: {our_feat.shape}, uni: {uni_feat.shape}, gigapath: {giga_feat.shape}, pathoduet: {pathoduet_feat.shape}, gpfm: {gpfm_feat.shape}")
        #all_feat_ours.append (our_feat)
        all_feat_giga.append (giga_feat[:,:16,:].reshape(-1, 1536))
        all_feat_giga.append (giga_feat[:,16:,:].reshape(-1, 1536))
        #all_feat_pathoduet.append (pathoduet_feat)
        #all_feat_gpfm.append (gpfm_feat)
        #all_feat_uni.append (uni_feat)
        #all_label_our.append ([ind]*len(our_feat))
        all_label_giga.append ([gt]*len(giga_feat[:,:16,:].reshape(-1, 1536)))#len(giga_feat))
        all_label_giga.append ([gt+2]*len(giga_feat[:,16:,:].reshape(-1, 1536)))
        #all_label_uni.append ([ind]*len(uni_feat))
        #all_label_pathoduet.append ([ind]*len(pathoduet_feat))
        #all_label_gpfm.append ([ind]*len(gpfm_feat))
        all_label.append ([gt,ind])
        slide_list.append(slide)
    except: continue
df = pd.concat([pd.DataFrame(slide_list), pd.DataFrame(all_label)], axis =1)
df.columns = ['slide', 'label', 'slide#']
print (df)
df.to_csv(f"{save_dir}plot{ii}_fold{fold}_{task}.csv", index=None)
all_feat_ours_ = np.concatenate(all_feat_ours)
all_feat_giga_ = np.concatenate(all_feat_giga)
all_feat_pathoduet_ = np.concatenate(all_feat_pathoduet)
all_feat_uni_ = np.concatenate(all_feat_uni)
all_feat_gpfm_ = np.concatenate(all_feat_gpfm)
all_label_our_ = np.concatenate(all_label_our)
all_label_giga_ = np.concatenate(all_label_giga)
all_label_uni_ = np.concatenate(all_label_uni)
all_label_pathoduet_ = np.concatenate(all_label_pathoduet)
all_label_gpfm_ = np.concatenate(all_label_gpfm)
# print (f' our: {all_feat_ours_.shape}, giga: {all_feat_giga_.shape}, uni: {all_feat_uni_.shape}, pathoduet: {all_feat_pathoduet_.shape}, gpfm: {all_feat_gpfm_.shape}')
label_colors = sns.color_palette("tab10")
n_components = 2
tsne = TSNE(n_components)
print ('Ploting for ours model')
tsne_our = tsne.fit_transform(all_feat_ours_)
tsne_our_df = pd.DataFrame({'tsne_1': tsne_our[:,0], 'tsne_2': tsne_our[:,1], 'label': all_label_our_})
fig_our, ax_our = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label',  data=tsne_our_df , ax=ax_our, s=8, palette=label_colors)
ax_our.set_aspect('equal')
_, legend_labels= ax_our.get_legend_handles_labels()#legend_labels = ['Class 1', 'Class 2']
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[i], markersize=8) for i in range(len(label_colors))]
ax_our.legend(handles=handles, labels=legend_labels, bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.0)
ax_our.axis('off')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f'{save_dir}/plot{ii}_ours_{task}_fold{fold}_slide{slide_num}.png')
plt.close('all')
print ('Ploting for Gigapath')
tsne_giga = tsne.fit_transform(all_feat_giga_)
tsne_giga_df = pd.DataFrame({'tsne_1': tsne_giga[:,0], 'tsne_2': tsne_giga[:,1], 'label': all_label_giga_})
fig_giga, ax_giga = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label',  data=tsne_giga_df , ax=ax_giga, s=8, palette=label_colors)
ax_giga.set_aspect('equal')
_, legend_labels= ax_giga.get_legend_handles_labels()#legend_labels = ['Class 1', 'Class 2']
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[i], markersize=8) for i in range(len(label_colors))]
ax_giga.legend(handles=handles, labels=legend_labels, bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.0)
ax_giga.axis('off')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f'{save_dir}/plot{ii}_giga_{task}_fold{fold}_slide{slide_num}.png')
plt.close('all')
print ('Ploting for UNI')
tsne_uni = tsne.fit_transform(all_feat_uni_)
tsne_uni_df = pd.DataFrame({'tsne_1': tsne_uni[:,0], 'tsne_2': tsne_uni[:,1], 'label': all_label_uni_})
fig_uni, ax_uni = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label',  data=tsne_uni_df , ax=ax_uni, s=8, palette=label_colors)
ax_uni.set_aspect('equal')
_, legend_labels= ax_uni.get_legend_handles_labels()#legend_labels = ['Class 1', 'Class 2']
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[i], markersize=8) for i in range(len(label_colors))]
ax_uni.legend(handles=handles, labels=legend_labels, bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.0)
ax_uni.axis('off')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f'{save_dir}/plot{ii}_uni_{task}_fold{fold}_slide{slide_num}.png')
plt.close('all')
print ('Ploting for Pathoduet')
tsne_pathoduet = tsne.fit_transform(all_feat_pathoduet_)
tsne_pathoduet_df = pd.DataFrame({'tsne_1': tsne_pathoduet[:,0], 'tsne_2': tsne_pathoduet[:,1], 'label': all_label_pathoduet_})
fig_pathoduet, ax_pathoduet = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label',  data=tsne_pathoduet_df , ax=ax_pathoduet, s=8, palette=label_colors)
ax_pathoduet.set_aspect('equal')
_, legend_labels= ax_pathoduet.get_legend_handles_labels()#legend_labels = ['Class 1', 'Class 2']
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[i], markersize=8) for i in range(len(label_colors))]
ax_pathoduet.legend(handles=handles, labels=legend_labels, bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.0)
ax_pathoduet.axis('off')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f'{save_dir}/plot{ii}_pathoduet_{task}_fold{fold}_slide{slide_num}.png')
plt.close('all')
print ('Ploting for GPFM')
tsne_gpfm = tsne.fit_transform(all_feat_gpfm_)
tsne_gpfm_df = pd.DataFrame({'tsne_1': tsne_gpfm[:,0], 'tsne_2': tsne_gpfm[:,1], 'label': all_label_gpfm_})
fig_gpfm, ax_gpfm = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label',  data=tsne_gpfm_df , ax=ax_gpfm, s=8, palette=label_colors)
ax_gpfm.set_aspect('equal')
_, legend_labels= ax_gpfm.get_legend_handles_labels()#legend_labels = ['Class 1', 'Class 2']
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=label_colors[i], markersize=8) for i in range(len(label_colors))]
ax_gpfm.legend(handles=handles, labels=legend_labels, bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0.0)
ax_gpfm.axis('off')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(f'{save_dir}/plot{ii}_gpfm_{task}_fold{fold}_slide{slide_num}_.png')
plt.close('all')

