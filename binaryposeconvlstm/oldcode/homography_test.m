
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



%% Szielski Book
img = imread('peppers.png');

FOV = 90;
theta = 45;
d = 500;

width = size(img, 2);
height = size(img, 1);
a = width / height;

f = (width / 2) / tan(FOV * 0.5 * pi/180);


W = width - 1;
H = height - 1;

tx = 0;
ty = 0;
tz = 0;

R_t = [cosd(theta), 0, -sind(theta), tx;
       0,           1, 0,            ty;
       sind(theta), 0, cosd(theta),  tz;
       0,           0, 0,            1];
   

cx = W / 2;
cy = H / 2;
   
K = [f, 0, cx, 0;
     0, a*f, cy, 0;
     0, 0, 1,  0;
     0, 0, 0,  1];
 
K_inv = K^(-1);
K0 = K;
K1 = K;

P0 = K0 * eye(4, 4);
P1 = K1 * R_t;

M10 = P1 * P0^(-1);
homography = M10(1 : 3, 1 : 3);

% https://stackoverflow.com/questions/6606891/opencv-virtually-camera-rotating-translating-for-birds-eye-view


A1 = [1, 0, -cx;
      0, 1, -cy;
      0, 0, 0;
      0, 0, 1];
A2 = [f, 0, cx, 0;
      0, a*f, cy, 0;
      0, 0, 1, 0];
  
R = R_t;
T = [1, 0, 0, 0;
     0, 1, 0, 0;
     0, 0, 1, d;
     0, 0, 0, 1];
 
homography = A2 * T * R * A1;

rect = [0, W, W, 0;
        0, 0, H, H;
        1, 1, 1, 1];
scatter(rect(1, :), rect(2, :));
hold on;



rect2 = homography * rect;
rect2 = rect2 ./ repmat(rect2(3, :), 3, 1);
   
scatter(rect2(1, :), rect2(2, :));


t = maketform('projective', homography');
warped = imtransform(img, t);

figure;
imshow(warped);


