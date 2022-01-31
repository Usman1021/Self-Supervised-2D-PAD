
function H = cvexEstStabilizationTform(leftI,rightI,ptThresh)

if nargin < 3 || isempty(ptThresh)
    ptThresh = 0.1;
end
leftI = rgb2gray(leftI);
rightI = rgb2gray(rightI);
% Collect sparse Points from Frame
 pointsA = detectSURFFeatures(leftI);
 pointsB = detectSURFFeatures(rightI);
 % Select Correspondences Between Points
[featuresA, pointsA] = extractFeatures(leftI,pointsA);
[featuresB, pointsB] = extractFeatures(rightI,pointsB);
indexPairs = matchFeatures(featuresA, featuresB);
pointsA = pointsA(indexPairs(:, 1), :);
pointsB = pointsB(indexPairs(:, 2), :);
% Estimating Transform
[tform, ~, ~, status] = estimateGeometricTransform(pointsB, pointsA, 'affine');
H = tform.T;
end