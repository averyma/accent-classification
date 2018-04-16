clc
clear

% load mfcc feature
mfcc = load('mfcc_feature.mat');
mfcc = mfcc.utterance;
mfcc = cell2mat(mfcc(3))';
% load mfcs feature
mfsc = load('mfsc_feature.mat');
mfsc = mfsc.utterance;
mfsc = cell2mat(mfsc(3))';
% load corresponding audio file
file = '/Users/ama/git-repo/accent-classification/cv-valid-test/sample-000022.mp3';
[sample,fs] = audioread(file);

% Plot MFCC over time
figure('Position', [30 100 800 200], 'PaperPositionMode', 'auto', ... 
     'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
imagesc( [1:size(mfcc,2)], [0:size(mfcc,1)-1], mfcc );
colormap('jet')
axis( 'xy' );
xlabel( 'Frame index' ); 
ylabel( 'Cepstrum index' );
title( 'Mel Frequency Cepstral Coefficient' );

% Plot MFSC over time
figure('Position', [30 100 800 200], 'PaperPositionMode', 'auto', ... 
     'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
imagesc( [1:size(mfsc,2)], [0:size(mfsc,1)-1], mfsc );
colormap('jet')
axis( 'xy' );
xlabel( 'Frame index' ); 
ylabel( 'Spectral index' );
title( 'Mel Frequency Spectral Coefficients' );

% plot audio signal
figure('Position', [30 100 800 200], 'PaperPositionMode', 'auto', ... 
     'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
plot([0:length(sample)-1],sample); xlabel('Samples'); ylabel('Amplitude');title('Audio Signal')

