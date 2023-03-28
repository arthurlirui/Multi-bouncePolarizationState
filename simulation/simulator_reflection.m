function S_r= simulator_reflection(filename)
%% Initial Mosic Data 
rootpath = '/home/qius0a/data/VMV_GroundTruth2018/mosic_images/color/';
%filename = 'ball'; % changes 
I_0  = imread(fullfile(rootpath, [filename, '_0.png']));
I_45 = imread(fullfile(rootpath, [filename, '_45.png']));
I_90 = imread(fullfile(rootpath, [filename, '_90.png']));
I_135= imread(fullfile(rootpath, [filename, '_135.png']));
%% Demosaic
I_0  = im2double(demosaic(I_0,'grbg')); %bggr
I_45 = im2double(demosaic(I_45,'grbg'));
I_90 = im2double(demosaic(I_90,'grbg'));
I_135= im2double(demosaic(I_135,'grbg'));
%% Stokes Vector
s0 = 1/2 * (I_0 + I_45 + I_90 + I_135);
s1 = I_0 - I_90;
s2 = I_45- I_135;
% Rotation
s0 = rot90(flip(s0, 2), 45);
s1 = rot90(flip(s1, 2), 45);
s2 = rot90(flip(s2, 2), 45);
S  = [s0; s1; s2];             
%% 
for d = 8 : 0.5 : 10  % thickness of glass
    for theta_i = 1 : 15 : 90
        theta_t = asind(sind(theta_i)/n);
        delta_x = 2*d*tand(theta_t)*sind(theta_t); % pixel translation
        %% Orthogonal Components
        %  reflection
        R1_para = ((tand(theta_i - theta_t))^2)/((tand(theta_i + theta_t))^2);
        R1_perp = ((sind(theta_i - theta_t))^2)/((sind(theta_i + theta_t))^2);
        T1_para = (sind(2*theta_i)*sind(2*theta_t))/((sind(theta_i + theta_t))^2 * (cosd(theta_i - theta_t))^2);
        T1_perp = (sind(2*theta_i)*sind(2*theta_t))/((sind(theta_i + theta_t))^2);
        r = (R1_para + R1_perp)/2;
        r_= (R1_para - R1_perp)/2;
        t = (T1_para + T1_perp)/2;
        t_= (T1_para - T1_perp)/2;

        R1= [r, r_, 0; r_, r, 0; 0, 0, sqrt(R1_para*R1_perp)];
        T1= [t, t_, 0; t_, t, 0; 0, 0, sqrt(T1_para*T1_perp)];
        % transmission        
        R2_para = ((tand(theta_t - theta_i))^2)/((tand(theta_t + theta_i))^2);
        R2_perp = ((sind(theta_t - theta_i))^2)/((sind(theta_t + theta_i))^2);
        T2_para = (sind(2*theta_t)*sind(2*theta_i))/((sind(theta_t + theta_i))^2 * (cosd(theta_t - theta_i))^2);
        T2_perp = (sind(2*theta_t)*sind(2*theta_i))/((sind(theta_t + theta_i))^2);       
        r2 = (R2_para + R2_perp)/2;
        r2_= (R2_para - R2_perp)/2;
        t2 = (T2_para + T2_perp)/2;
        t2_= (T2_para - T2_perp)/2; 

        R2= [r2, r2_, 0; r2_, r2, 0; 0, 0, sqrt(R2_para*R2_perp)];
        T2= [t2, t2_, 0; t2_, t2, 0; 0, 0, sqrt(T2_para*T2_perp)];
        % multi-bounce is 3
        S_r = R1*S + ...
              T2*R2*T1*S*(imtranslate(S,[delta_x, 0],'FillValues',255)) + ...
              T2*(R2^3)*T1*S*(imtranslate(S,[2*delta_x, 0],'FillValues',255));    
    end
end