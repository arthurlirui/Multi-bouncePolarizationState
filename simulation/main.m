

rootpath = '/home/lir0b/Code/TransparenceDetection/src/perceptual-reflection-removal/synthetic';
flist_r = dir(fullfile(rootpath, 'reflection_layer', '*.jpg'));
flist_t = dir(fullfile(rootpath, 'transmission_layer', '*.jpg'));
numsample = 1000;
fidx_r = randi([1, numsample], [1,numsample]);
fidx_t = randi([1, numsample], [1,numsample]);
n = 1.5; d=10; mirror_ref = 0.3; theta_i = 50;
default_params;
for i = 1:numsample
    filename_r = flist_r(fidx_r(i)).name;
    filename_t = flist_t(fidx_t(i)).name;
    disp(i)
    Ir = im2double(imread(fullfile(rootpath, 'reflection_layer', filename_r)));
    It = im2double(imread(fullfile(rootpath, 'transmission_layer', filename_t)));
    default_params;
    params.savefolder = [filename_r(1:end-4), '_', filename_t(1:end-4)];
    mkdir(fullfile(params.savepath, params.savefolder));
    %[Ir_polar] = synthetic_ghost(Ir, It, params);
    
    %% polarized ray tracing
    Ir_perp = 1/2*params.ref_coef*Ir; Ir_para = 1/2*params.ref_coef*Ir;
    It_perp = 1/2*params.trans_coef*It; It_para = 1/2*params.trans_coef*It;
    phi_perp = 0; phi_para = 90;
    for i = 1:length(params.thetalist)
        theta_i = round(params.thetalist(i));
        
        %[I_perp_out, I_para_out] = polar_engine(I_perp, I_para, theta_i, n, d);
        [Ir_perp_out, Ir_para_out] = polar_engine(Ir_perp, Ir_para, theta_i, n, d, mirror_ref, 1); 

        [It_perp_out, It_para_out] = polar_engine(It_perp, It_para, theta_i, n, d, mirror_ref, 0); 
        Iref=Ir_perp_out+Ir_para_out; 
        Itrans=It_perp_out+It_para_out; 
        Iout = Iref+Itrans;
        savename = ['S0', '_theta_', num2str(theta_i), '_out.jpg'];
        imwrite(Iout, fullfile(params.savepath, params.savefolder, savename));
        savename = ['S0', '_theta_', num2str(theta_i), '_ref.jpg'];
        imwrite(Iref, fullfile(params.savepath, params.savefolder, savename));
        savename = ['S0', '_theta_', num2str(theta_i), '_trans.jpg'];
        imwrite(Itrans, fullfile(params.savepath, params.savefolder, savename));
        
        savesubfolder = 'polar';
        savepath_phi = fullfile(params.savepath, params.savefolder, savesubfolder);
        if ~exist(savepath_phi), mkdir(savepath_phi); end
        
        for j = 1:length(params.philist)
            phi = params.philist(j);
            Ir_filter = Ir_perp_out*cosd(phi - phi_perp)^2+Ir_para_out*cosd(phi - phi_para)^2;
            It_filter = It_perp_out*cosd(phi - phi_perp)^2+It_para_out*cosd(phi - phi_para)^2;
            I_filter = Ir_filter+It_filter;
            savename = ['S0', '_theta_', num2str(theta_i), '_phi_', num2str(phi), '_filter.jpg'];
            imwrite(I_filter, fullfile(params.savepath, params.savefolder, savesubfolder, savename));
            savename = ['S0', '_theta_', num2str(theta_i), '_phi_', num2str(phi), '_filter_r.jpg'];
            imwrite(Ir_filter, fullfile(params.savepath, params.savefolder, savesubfolder, savename));
            savename = ['S0', '_theta_', num2str(theta_i), '_phi_', num2str(phi), '_filter_t.jpg'];
            imwrite(It_filter, fullfile(params.savepath, params.savefolder, savesubfolder, savename));
        end
    end
    
end


