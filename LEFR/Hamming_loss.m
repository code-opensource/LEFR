function HammingLoss=Hamming_loss(Pre_Labels,test_target)
%Computing the hamming loss 计算海明损失
%Pre_Labels: the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(j,i)=1, otherwise Pre_Labels(j,i)=-1
%Pre_Labels：分类器的预测标签，如果第i个实例属于第j类，Pre_Labels（j，i）= 1，否则Pre_Labels（j，i）= - 1
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1
%test_target：测试实例的实际标签，如果第i个实例属于第j个类，test_target（j，i）= 1，否则test_target（j，i）= - 1
    [num_class,num_instance]=size(Pre_Labels);
    miss_pairs=sum(sum(Pre_Labels~=test_target));
    HammingLoss=miss_pairs/(num_class*num_instance);