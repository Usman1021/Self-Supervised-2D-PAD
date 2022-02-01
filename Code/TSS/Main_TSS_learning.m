%% This research is made available to the research community.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% If you are using this code please cite the following paper:                                                                                      %
% Muhammad, Usman, Zitong Yu, and Jukka Komulainen. "Self-supervised 2D face presentation attack detection via temporal sequence sampling." (2021). %
% ..................................................................................................................................................%
%% For any problem in running the code, please contact me through following emails: usman@mail.bnu.edu.cn  or muhammad.usman@oulu.fi                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% TSS learning for videos
% Place inthe folder where videos exist and provide the video format as input
x = dir(input);
[length temp] = size(x);
for k=1:length
    videoname = x(k).name;
    videoFileReader = vision.VideoFileReader(videoname);

    numFrames = 0;
    while ~isDone(videoFileReader)
        step(videoFileReader);
        numFrames = numFrames + 1;
    end

    reset(videoFileReader);                   
    movMean = step(videoFileReader);
    imgB = movMean;
    imgBp = imgB;
    correctedMean = imgBp;
    % set the video length 
    range = 15:15:numFrames;
    Hcumulative = eye(3);
    for i=1:size(range,2)
        ii=range(i);
        ref=ii;
        while ~isDone(videoFileReader) && ii < ref + 15
            imgA = imgB; 
            imgAp = imgBp;
            imgB = step(videoFileReader);
         % Estimate transform from frame A to frame B, and fit as an s-R-t
            H = cvexEstStabilizationTform(imgA,imgB);
            HsRt = cvexTformToSRT(H);
            Hcumulative = HsRt * Hcumulative;
            img = imwarp(imgB,affine2d(Hcumulative),'OutputView',imref2d(size(imgB)));
             correctedMean = correctedMean + img;
            ii = ii+1;
        end
        correctedMean = correctedMean/(15);
        imwrite(correctedMean,strcat( string(k), string(i),'.jpg'));
    end
end
