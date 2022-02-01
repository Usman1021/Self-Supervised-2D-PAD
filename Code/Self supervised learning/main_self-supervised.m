Clear all
close all
clc

%% This research is made available to the research community.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you are using this code please cite the following paper:                                                                                      %
% Muhammad, Usman, Zitong Yu, and Jukka Komulainen. "Self-supervised 2D face presentation attack detection via temporal sequence sampling." (2021). % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Train the model for Self-supervised Learning
% Follow the instructions given by the datasets to split real and fake videos. Make two folders named as “real and “attack”. 
% Put the videos into these folders and train the model for face anti-spoofing. %

% Input the folder link of your training data
rootFolder = fullfile("input");
categories  = {'15','30','45','60'};
trainingset = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
% Input the folder link of your Development set
rootFolder = fullfile("input");
categories  = {'15','30','45','60'};
developmentset = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

%% Pretext task learning
% Input for ResNet-101 model
net = resnet101;
% Specify the layers to remove
net.Layers(1);
 lgraph = layerGraph(net);
 layersToRemove = {
     'fc1000'
     'prob'
     'ClassificationLayer_predictions'
     };
 lgraph = removeLayers(lgraph, layersToRemove);
 numClasses = 4;
 % Specify new layers
 newLayers = [
     fullyConnectedLayer(numClasses, 'Name', 'rcnnFC')
     softmaxLayer('Name', 'rcnnSoftmax')
     classificationLayer('Name', 'rcnnClassification')
     ];
 lgraph = addLayers(lgraph, newLayers);
 lgraph = connectLayers(lgraph,  'pool5' , 'rcnnFC');
 % Specify the learning rate
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
% train the model without labels
 net1 = trainNetwork(trainingset,lgraph,options);
 %%
 % Input for Target set
rootFolder = fullfile(input);
categories  = {'real','attack'};
trainingset = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
%Input for development set 
rootFolder = fullfile(input);
categories  = {'real','attack'};
developmentset = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');  
% Input for Testing set 
rootFolder = fullfile(input);
categories = {'real','attack'};
testingsetdata = imageDatastore(fullfile(rootFolder, categories),  'IncludeSubfolders',true, ...
 'LabelSource','foldernames');
% specify the training labels, devlp and testing labels
trainingLabels = trainingset.Labels;
developmentlabel = developmentset.Labels;
testinglabel = testingsetdata.Labels;
% Input of already trained model during Pretext task learning
net1.Layers(1);
 lgraph = layerGraph(net1);
 layersToRemove = {
     'rcnnFC'
     'rcnnSoftmax'
     'rcnnClassification'
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
% Setting the options
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
% train the model 
net2 = trainNetwork(trainingset,lgraph,options);
% extract features for BiLSTM input 
featureLayer = 'pool5' ;
trainingFeatures1 = activations(net2, trainingset, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
developmentFeatures1 = activations(net2,developmentset, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
testingFeatures1 = activations(net2, testingsetdata, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% Training for BiLSTM network
rng(1);
trainf = {};
trainf{end+1} =  trainingFeatures1;

trainlabl = {};
trainlabl{end+1} = trainingLabels';

train1 = {};
train1{end+1} = developmentFeatures1;
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
options = trainingOptions('adam', ...
     'ExecutionEnvironment','gpu', ... 
       'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{train1,train2}, ...
    'ValidationFrequency',30, ...
     'SequenceLength','longest', ...
    'Plots','training-progress', ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
% BiLSTM training
lstm1 = trainNetwork(trainf',trainlabl,layers,options);
[~, devlp_scores1] = classify(lstm1, developmentFeatures1);
 numericLabels1 = grp2idx(developmentlabel);
 numericLabels1(numericLabels1==2)= -1;
 numericLabels1(numericLabels1==1)= 1;
 devlpscores1 =devlp_scores1';
 [TPR,TNR,Info]=vl_roc(numericLabels1,devlpscores1(:,1));
 % compute EER
 EER = Info.eer*100
 threashold = Info.eerThreshold; 
 [~, test_scores1] = classify(lstm1, testingFeatures1);
 testscores1 = test_scores1';
 numericLabels = grp2idx(testinglabel);
 numericLabels(numericLabels==2)= -1;
 numericLabels(numericLabels==1)= 1;
 real_scores1 =  testscores1(numericLabels==1);
 attack_scores2 =  testscores1(numericLabels==-1);
 FAR = sum(attack_scores2>threashold) / numel(attack_scores2)*100;
 FRR = sum(real_scores1<=threashold) / numel(real_scores1)*100;
 % Compute HTER
 HTER = (FAR+FRR)/2;
