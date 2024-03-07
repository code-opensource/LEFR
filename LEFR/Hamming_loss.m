function HammingLoss=Hamming_loss(Pre_Labels,test_target)
%Computing the hamming loss ���㺣����ʧ
%Pre_Labels: the predicted labels of the classifier, if the ith instance belong to the jth class, Pre_Labels(j,i)=1, otherwise Pre_Labels(j,i)=-1
%Pre_Labels����������Ԥ���ǩ�������i��ʵ�����ڵ�j�࣬Pre_Labels��j��i��= 1������Pre_Labels��j��i��= - 1
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1
%test_target������ʵ����ʵ�ʱ�ǩ�������i��ʵ�����ڵ�j���࣬test_target��j��i��= 1������test_target��j��i��= - 1
    [num_class,num_instance]=size(Pre_Labels);
    miss_pairs=sum(sum(Pre_Labels~=test_target));
    HammingLoss=miss_pairs/(num_class*num_instance);