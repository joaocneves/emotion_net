data = load('experiments\DATASET_MNIST_NUM_EPOCHS_100_BATCH_SIZE_16_TYPE_baseline\debug_data_epoch_0.mat', ...
    'intermediate_output_val', 'samples_label_val', 'loss_per_sample_val');
feat = data.intermediate_output_val;
feat_label = data.samples_label_val;
loss = data.loss_per_sample_val;

[~,feat_pca] = pca(feat);

feat_tsne = tsne(feat_pca);%, 'Algorithm', 'exact');

gscatter(feat_tsne(:,1), feat_tsne(:,2), feat_label)

Idx = knnsearch(feat_tsne,feat_tsne, 'K', 21);
pred = feat_label(Idx);
gt = repmat(pred(:,1),1,20);
pred = pred(:,2:end);

C = zeros(size(pred,1), 10*10);
for i = 1:size(pred,1)
    C(i,:)=reshape(confusionmat(gt(i,:),pred(i,:),'Order',0:9),[],1);
end

[cluster_idx, centroid] = kmeans(C,12);

for i = 1:13
    centroid_i = reshape(centroid(i,:),10,10);
    figure,gscatter(feat_tsne(cluster_idx==i,1),feat_tsne(cluster_idx==i,2),feat_label(cluster_idx==i))
    figure,hist(loss(cluster_idx==i))
end

