%% This research is made available to the research community.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you are using this code please cite the following paper:                                                                                      %
% Muhammad, Usman, Zitong Yu, and Jukka Komulainen. "Self-supervised 2D face presentation attack detection via temporal sequence sampling." (2021). % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% For any problem in running the code, please contact me through following emails: usman@mail.bnu.edu.cn  or muhammad.usman@oulu.fi
%% Train the Model for Supervised Learning
% Follow the instructions given by the datasets to split real and fake videos. Make two folders named as “real and “attack”. 
% Put the videos into these folders and train the model for face anti-spoofing. 

%% Input the folder link of your training data
rootFolder = fullfile(input);
categories = {'real', 'attack'};
Train = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');  
trainingLabels = Train.Labels;
%% % Input the folder link of your development data
rootFolder = fullfile(input);
categories = {'real', 'attack'};
developmentset = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames'); 
developmentlabel = developmentset.Labels;
%% % Input the folder link of your Testing data
rootFolder = fullfile(input);
categories = {'real', 'attack'};
testset = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames'); 
testlabels = testset.Labels;
%% CNN MODEL
% Input for CNN model
net = resnet101;
% Specify the layers to remove
net.Layers(1)
 lgraph = layerGraph(net);
 layersToRemove = {
     'fc1000'
     'prob'
     'ClassificationLayer_predictions'
     };
 lgraph = removeLayers(lgraph, layersToRemove);
 numClasses = 2;
 newLayers = [
     fullyConnectedLayer(numClasses, 'Name', 'rcnnFC')
     softmaxLayer('Name', 'rcnnSoftmax')
     classificationLayer('Name', 'rcnnClassification')
     ];
 lgraph = addLayers(lgraph, newLayers);
 lgraph = connectLayers(lgraph,  'pool5' , 'rcnnFC');
miniBatchSize = 32;
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ... 
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',0.0001, ...
    'Shuffle','every-epoch', ...   
    'ValidationData',developmentset, ...
   'ValidationFrequency',30, ...
    'Verbose',false, ...
      'Plots','training-progres', ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
% Train the model supervised learning
net = trainNetwork(Train,lgraph,options);
% extract features for BiLSTM
featureLayer = 'pool5';
trainingFeatures = activations(net, Train, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
developmentFeatures = activations(net,developmentset, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
testingFeatures = activations(net, testset, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
%%
% Training for BiLSTM model
rng(1);
trainf = {};
trainf{end+1} =  trainingFeatures;

trainlabl = {};
trainlabl{end+1} = trainingLabels';

train1 = {};
train1{end+1} = developmentFeatures;
% 
train2 = {};
train2{end+1} = developmentlabel';
numFeatures =2048;
 numHiddenUnits =100;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
         bilstmLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer','he')
     fullyConnectedLayer(numClasses,'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];
% Define learning options
options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress', ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
% Train BilSTM
lstm = trainNetwork(trainf',trainlabl,layers,options);
[~, devlp_scores2] = classify(lstm, developmentFeatures);
 numericLabels1 = grp2idx(developmentlabel);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores2 =devlp_scores2';
 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscores2(:,1));
 % compute EER
 EER = Info.eer*100
 threashold = Info.eerThreshold;
 [~, test_scores2] = classify(lstm, testingFeatures);
 testscores2 = test_scores2';
 numericLabels = grp2idx(testlabels);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores2(numericLabels==1);
 attack_scores2 =  testscores2(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100
 % Compute HTER
 HTER = (FAR+FRR)/2
