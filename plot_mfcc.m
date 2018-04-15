clc
clear

mfsc = load('norm_feature.mat');
mfsc = mfsc.utterance;
% mfsc = reshape(mfsc(450,:,:),[size(mfsc,2) size(mfsc,3)]);
mfsc = cell2mat(mfsc(1))';
% mfsc = mfsc(:,1:10)
% mean(mfsc,1)

% Plot cepstrum over time
figure('Position', [30 100 800 200], 'PaperPositionMode', 'auto', ... 
     'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 
% imagesc( [1:size(mfsc,2)], [0:size(mfsc,1)-1], (mfsc-mean(mfsc,2))./std(mfsc,2) );
imagesc( [1:size(mfsc,2)], [0:size(mfsc,1)-1], mfsc );
colormap('jet')
axis( 'xy' );
xlabel( 'Frame index' ); 
ylabel( 'Cepstrum index' );
title( 'Mel frequency cepstrum' );
