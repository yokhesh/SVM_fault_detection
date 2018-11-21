%train correct
train_correct = load('training_correct.txt');
train_faulty = load ('training_faulty.txt');
test_correct = load('testing_correct.txt');
test_faulty = load('testing_faulty.txt');

training = [train_correct;train_faulty];
testing = [test_correct;test_faulty];
%training data preprocess

mean_train = mean(training');
st_train = std(training');


var_train = var(training');
rms_train = rms(training');
p2p_train = peak2peak(training');

rsq_train = rssq(training');


max_train = max(training');
sum_train = sum(training');
training = [mean_train;st_train;var_train;rms_train;p2p_train;rsq_train;max_train;sum_train];
%testing data preprocess
mean_test = mean(testing');
st_test = std(testing');

var_test = var(testing');
rms_test = rms(testing');
p2p_test = peak2peak(testing');
sum_test = sum(testing');
rsq_test = rssq(testing');
max_test = max(testing');
testing = [mean_test;st_test;var_test;rms_test;p2p_test;rsq_test;max_test;sum_test]';

training_labels = [ones((size(train_correct,1)),1);-1*ones((size(train_faulty,1)),1)];
testing_labels = [ones((size(test_correct,1)),1);-1*ones((size(test_faulty,1)),1)];
%Scaling
% min_traindata = min(training);
% range = max(training) - min_traindata;
% testing = (testing - repmat(min_traindata, size(testing, 1), 1)) ./ repmat(range, size(testing, 1), 1);
% training = (training - repmat(min_traindata, size(training, 1), 1)) ./ repmat(range, size(training, 1), 1);

training = (training./100)';
 testing = testing./100;
% %linear
% disp(' Linear Support Vector Machine');
training_s = svmtrain(training_labels,training, '-s 1 -t 0 -b 0 -h 0 -g 1 -c 1');
[predicted_value]=svmpredict(testing_labels,testing,training_s);