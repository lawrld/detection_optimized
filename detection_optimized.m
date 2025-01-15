% load pre-trained network resnet-18
net = resnet18; % load resnet-18 
lgraph = layerGraph(net);

% modify network for binary classification
numClasses = 2; % human and non-human
newFc = fullyConnectedLayer(numClasses, 'Name', 'fc_binary');
newOutput = classificationLayer('Name', 'output');

lgraph = replaceLayer(lgraph, 'fc1000', newFc); % replace final fully connected layer 
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newOutput); % replace classification layer 

% load dataset
dataFolder = 'C:\Users\Lance Avist\Downloads\Optional Bonus Project\Optional Bonus Project\archive\human detection dataset';
imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% map folder names to labels
imds.Labels = categorical(imds.Labels, {'0', '1'}, {'Non-Human', 'Human'});

% split dataset into training and validation
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% resize images and augment data
inputSize = net.Layers(1).InputSize(1:2);

% augmentation configuration
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5], ...
    'RandXScale', [0.9, 1.1], ...
    'RandYScale', [0.9, 1.1]);

augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);
augmentedValidation = augmentedImageDatastore(inputSize, imdsValidation);

% training configurations 
options = trainingOptions('adam', ... % Use Adam optimizer
    'MiniBatchSize', 16, ... % Lower batch size
    'MaxEpochs', 20, ... % Increased epochs
    'InitialLearnRate', 1e-4, ... % Starting learning rate
    'LearnRateSchedule', 'piecewise', ... % Use piecewise schedule
    'LearnRateDropFactor', 0.1, ... % Drop learning rate by 10%
    'LearnRateDropPeriod', 5, ... % Drop learning rate every 5 epochs
    'ValidationData', augmentedValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress'); 

% Train the Network
trainedNet = trainNetwork(augmentedTrain, lgraph, options);

% Save the Trained Model
save('trainedHumanDetector_Augmented.mat', 'trainedNet');
disp('Model trained and saved with augmentation and improved settings.');

%% Load and Test the Model
% Load the Trained Network
load('trainedHumanDetector_Augmented.mat', 'trainedNet');

% Use Validation Data for Testing
augmentedValidation = augmentedImageDatastore(inputSize, imdsValidation); % resize validation set to match input size 
% Classify Validation Images
[predictedLabels, scores] = classify(trainedNet, augmentedValidation);

% Actual Labels
actualLabels = imdsValidation.Labels;

% Calculate Accuracy
accuracy = mean(predictedLabels == actualLabels) * 100;
disp(['Validation Accuracy: ', num2str(accuracy), '%']);

%% Test on img5.jpg with Smaller Bounding Boxes
% Load the Test Image
testImagePath = 'C:\Users\Lance Avist\Downloads\Optional Bonus Project\Optional Bonus Project\img5.jpg';
testImage = imread(testImagePath);

% Preprocess the Test Image
[imgHeight, imgWidth, ~] = size(testImage);

% Define Smaller Window Size
windowSize = round(inputSize * 0.50); % Reduce the window size to 75% of the input size
windowHeight = windowSize(1);
windowWidth = windowSize(2);

% Set Sliding Window Parameters
stride = 64; % Stride for the sliding window
threshold = 0.8; % Confidence threshold for detection

% Initialize results storage
boundingBoxes = [];
labels = [];
scores = [];

% sliding window 
for y = 1:stride:(imgHeight - windowHeight)
    for x = 1:stride:(imgWidth - windowWidth)
        % extract sub-region 
        window = testImage(y:y + windowHeight - 1, x:x + windowWidth - 1, :);

        % resize window 
        resizedWindow = imresize(window, inputSize);

        % classify sub-region 
        [label, score] = classify(trainedNet, resizedWindow);

        % store result if confidence is above threshold 
        if max(score) > threshold
            boundingBoxes = [boundingBoxes; x, y, windowWidth, windowHeight]; %#ok<AGROW>
            labels = [labels; label]; 
            scores = [scores; max(score)]; 
        end
    end
end

% display original images with bounding boxes
figure;
imshow(testImage);
hold on;

% draw and annotate bounding boxes 
for i = 1:size(boundingBoxes, 1)
    rectangle('Position', boundingBoxes(i, :), 'EdgeColor', 'r', 'LineWidth', 2);
    text(boundingBoxes(i, 1), boundingBoxes(i, 2) - 10, ...
        sprintf('%s (%.2f%%)', char(labels(i)), scores(i) * 100), ...
        'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'black');
end

title('Human Detection Results');
hold off;

disp('Bounding box detection completed!');
