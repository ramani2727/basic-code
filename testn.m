clc;
clear all;
close all;

cd ds
%%  PRE PROCESS

[fn, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.dcm';'*.pgm';'*.gif'}, 'Pick a Image File');

im = imread(fn);

  figure,imshow(im);title('input');


I1 = imresize(im,[250,250]);

figure,imshow(I1);impixelinfo;title('resize');

J = imnoise(I1,'salt & pepper',0.02);

figure,imshow(J);title('noisy');

 K = medfilt2(J);
  
figure,imshow(K) ; 
title('median filter');


%%  PERFORMANCE

%MSE,PSNR ,SSIM,SNR measurement--1,
% % 
% [peaksnr, snr] = psnr(I1, K);
%   
% psnr=sprintf('\n The Peak-SNR value is %0.4f', peaksnr);
% msgbox(psnr);
% 
% snr=sprintf('\n The SNR value is %0.4f \n', snr);
% msgbox(snr)
% 
% err = immse(I1, K);
% mse=sprintf('\n The mean-squared error is %0.4f\n', err);
% msgbox(mse);
% 
% [ssimval,ssimmap] = ssim(K,I1)
% 
% figure,imshow(ssimmap,[])
% title(['Local SSIM Map with Global SSIM Value: ',num2str(ssimval)])
% 

%%  Enhance Contrast

I = imadjust(K,stretchlim(K));

 figure();
 imshow(I);impixelinfo;title('enhanced image');

I=label2rgb(I);


%%  K MEANS SEGMENTATION


im = double(I);
s_img = size(I);
r = im(:,:,1);
g = im(:,:,2);
b = im(:,:,3);

data_vecs = [r(:) g(:) b(:)];

k= 4;

[ idx C ] = kmeansK( data_vecs, k );

palette = round(C);

%Color Mapping

idx = uint8(idx);
outImg = zeros(s_img(1),s_img(2),3);
temp = reshape(idx, [s_img(1) s_img(2)]);
for i = 1 : 1 : s_img(1)
    for j = 1 : 1 : s_img(2)
        outImg(i,j,:) = palette(temp(i,j),:);
        
    end
end
figure,imshow(outImg);
cluster1 = zeros(size(r));
cluster2 = zeros(size(r));
cluster3 = zeros(size(r));
cluster4 = zeros(size(r));

cluster1(find(outImg(:,:,1)==palette(1,1))) = 1;
figure, 
subplot(2,2,1), imshow(cluster1);

cluster2(find(outImg(:,:,1)==palette(2,1))) = 1;

subplot(2,2,2), imshow(cluster2);

cluster3(find(outImg(:,:,1)==palette(3,1))) = 1;

 subplot(2,2,3), imshow(cluster3);

 cluster4(find(outImg(:,:,1)==palette(4,1))) = 1;
 
 subplot(2,2,4), imshow(cluster4);

 %%  GLCM AND STATISTICAL fetaures 

 cc=cluster1; 

glcms = graycomatrix(cc);
stats = graycoprops(glcms,'Contrast Correlation');

stats1 = graycoprops(glcms,'Energy Homogeneity');

c1=stats.Contrast;

c2=stats.Correlation;

e1=stats1.Energy;

h1=stats1.Homogeneity;

bw1=cc;
me=mean2(bw1);
st=std2(bw1);
va=var(var(double(bw1)));
sk=skewness(skewness(double(bw1)));
ku=kurtosis(kurtosis(double(bw1)));

%%
aa=cluster2;

glcms = graycomatrix(aa);
stats = graycoprops(glcms,'Contrast Correlation');

stats1 = graycoprops(glcms,'Energy Homogeneity');

c2=stats.Contrast;

cc2=stats.Correlation;

e2=stats1.Energy;

h2=stats1.Homogeneity;

me2=mean2(aa);
st2=std2(aa);
va2=var(var(double(aa)));
sk2=skewness(skewness(double(aa)));
ku2=kurtosis(kurtosis(double(aa)));
%%
bb=cluster3;

glcms = graycomatrix(bb);
stats = graycoprops(glcms,'Contrast Correlation');

stats1 = graycoprops(glcms,'Energy Homogeneity');

c3=stats.Contrast;

cc3=stats.Correlation;

e3=stats1.Energy;

h3=stats1.Homogeneity;

me3=mean2(bb);
st3=std2(bb);
va3=var(var(double(bb)));
sk3=skewness(skewness(double(bb)));
ku3=kurtosis(kurtosis(double(bb)));
%%
dd=cluster4;

glcms = graycomatrix(dd);
stats = graycoprops(glcms,'Contrast Correlation');

stats1 = graycoprops(glcms,'Energy Homogeneity');

c4=stats.Contrast;

cc4=stats.Correlation;

e4=stats1.Energy;

h4=stats1.Homogeneity;

me4=mean2(dd);
st4=std2(dd);
va4=var(var(double(dd)));
sk4=skewness(skewness(double(dd)));
ku4=kurtosis(kurtosis(double(dd)));


%% FEATURES

QF=[me st va sk ku c1 c2 e1 h1 me2 st2 va2 sk2 ku2 c2 cc2 e2 h2 me3 st3 va3 sk3 ku3 c3 cc3 e3 h3 me4 st4 va4 sk4 ku4 c4 cc4 e4 h4 ];

k=double(QF);

TF=k;

%% CLASSIFICATION

 load traintreenew.mat

 X=(x);

Y=[ones(10,1);2*ones(13,1);11*ones(6,1);12*ones(19,1);13*ones(16,1);14*ones(6,1);15*ones(4,1)]

Mdl = fitcdiscr(X,Y)

result=predict(Mdl,(TF));


if result == 1
    msgbox(' NORMAL');
elseif result == 2
    msgbox(' BENIGN');
elseif result == 11
    msgbox(' MALIGNANT-STAGE 1');
elseif result == 12
    msgbox(' MALIGNANT-STAGE 2');
elseif result == 13
    msgbox(' MALIGNANT-STAGE 3');
elseif result == 14
    msgbox(' MALIGNANT-STAGE 4');
elseif result == 15
    msgbox(' MALIGNANT-STAGE 5');
end


