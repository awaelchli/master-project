
img = imread('peppers.png');

imshow(img);


theta = 30;
% rot = [cosd(theta), 0, sind(theta);
%        0, 1, 0;
%        -sind(theta), 0, cosd(theta)];
   
rot = roty(theta);
   
transl = [1, 0, cos(-theta);
          0, 1, 0;
          0, 0, 1];
      
      
transl = eye(3,3);

rot = transl * rot; % Rx + t

height = 10;
width = 10;

rect = [0, width - 1,   width - 1,      0;
        0, 0,           height - 1,     height - 1;
        1, 1,           1,              1];

f = 1;

K = [ f*(width-1),       0,        (width - 1) / 2;
      0,             f*(height-1)  (height - 1) / 2;
      0,                 0,        1                   ];

K_inv = K^(-1);

% rect = [-0.5, -0.5, 0.5, 0.5;
%         0.5, -0.5, 0.5, -0.5;
%         1, 1, 1, 1];
    
scatter(rect(1, :), rect(2, :));
hold on;

H =  K * rot * K_inv;

rect2 = H * rect;
rect2 = rect2 ./ repmat(rect2(3, :), 3, 1);
   
scatter(rect2(1, :), rect2(2, :));
   
   
%%

% t = maketform('projective', H);
% warped = imtransform(img, t);

t = projective2d(H);
warped = imwarp(img, t);

figure;
imshow(warped);
