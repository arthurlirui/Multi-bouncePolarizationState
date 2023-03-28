%% Calculaing Unknown: I_perp, I_para, and fi_perp.


i = 1 : n;
% Total Intensity: I
I = 1/2 * Ir * ( R_perp(i) * cosd(fi  - fi_perp)^2 + R_para(i) * cosd(fi  - fi_para)^2 ); 

% Total Intensity: I_fi = I_perp * cos + I_para * sin
I_fi = I_perp * cosd(fi - fi_perp)^2 + I_para * sin(fi - fi_perp)^2;

% Three diff 'fi' observations of the 'same scene' : I_fi(1) 0 degree captured image...
I_fi = [];
for fi = 1 : 1 : 3 % 0, 45, 90;
I_fi = I_perp * cosd(fi - fi_perp)^2 + I_para * sin(fi - fi_perp)^2;
end