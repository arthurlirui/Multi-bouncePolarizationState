function [I_perp_out, I_para_out] = polar_engine(I_perp, I_para, theta_i, n, d, ref_mirror, is_ref)
    %ref_mirror = 0.3; % mirror type reflection
    num_bounce = 10;
    Ir_list = {};
    It_list = {};
    flag = 1; % low density to high density
    % from air to surface 1
    theta_in = theta_i;
    theta_out = asind(sind(theta_in)/n);  
    I_perp_cell = {}; I_para_cell = {};
    if(is_ref),
        %[R_perp, R_para, T_perp, T_para] = calc_RT(theta_in, theta_out);
        Ir_perp_out = I_perp*ref_mirror;  
        Ir_para_out = I_para*ref_mirror;  
        It_perp_out = I_perp*(1-ref_mirror);
        It_para_out = I_para*(1-ref_mirror);
        I_perp_cell{1} = Ir_perp_out;
        I_para_cell{1} = Ir_para_out;
        for i = 1:num_bounce,
            % bounce from surface 2 to surface 1 
            [R_perp, R_para, T_perp, T_para] = calc_RT(theta_out, theta_in);
            Ir_perp_out = It_perp_out*R_perp;
            Ir_para_out = It_para_out*R_para;
            
            %It_perp_out = It_perp_out*T_perp; % from surface 2 to air
            %It_para_out = It_para_out*T_para;

            % bounce from surface 1 to air
            %[R_perp, R_para, T_perp, T_para] = calc_RT(theta_out, theta_in);
            It_perp_out = Ir_perp_out*T_perp;
            It_para_out = Ir_para_out*T_para;
            
            Ir_perp_out = Ir_perp_out*R_perp; % new interior reflection
            Ir_para_out = Ir_para_out*R_para;
            

            %It_perp_out = Ir_perp_out;
            %It_para_out = Ir_para_out;

            It_perp_out = imtranslate(It_perp_out, [ 2*d*tand(theta_out), 0]);
            It_para_out = imtranslate(It_para_out, [ 2*d*tand(theta_out), 0]);
            I_perp_cell{i+1} = It_perp_out;
            I_para_cell{i+1} = It_para_out;
            
            % next round reflectio light
            It_perp_out = Ir_perp_out;
            It_para_out = Ir_para_out;
        end
    else
        % transmitted case
        %Ir_perp_out = I_perp*ref_mirror;  
        %Ir_para_out = I_para*ref_mirror;  
        %It_perp_out = I_perp*(1-ref_mirror);
        %It_para_out = I_para*(1-ref_mirror);
        [R_perp, R_para, T_perp, T_para] = calc_RT(theta_in, theta_out);
        % from surface 2 to surface 1
        I_perp_cell{1} = I_perp*T_perp;
        I_para_cell{1} = I_para*T_para;
        It_perp_out = I_perp*T_perp;
        It_para_out = I_para*T_para;
        for i = 1:num_bounce,
            % bounce from surface 1 to surface 2 
            [R_perp, R_para, T_perp, T_para] = calc_RT(theta_out, theta_in);
            Ir_perp_out = It_perp_out*R_perp;
            Ir_para_out = It_para_out*R_para;
            %It_perp_out = It_perp_out*T_perp;
            %It_para_out = It_para_out*T_para;

            % bounce from surface 2 to surface 1
            [R_perp, R_para, T_perp, T_para] = calc_RT(theta_out, theta_in);
            Ir_perp_out = Ir_perp_out*R_perp;
            Ir_para_out = Ir_para_out*R_para;
            %It_perp_out = Ir_perp_out*T_perp;
            %It_para_out = Ir_para_out*T_para;
            
            % from surface 1 to air
            It_perp_out = Ir_perp_out*T_perp;
            It_para_out = Ir_para_out*T_para;

            Ir_perp_out = Ir_perp_out*R_perp;
            Ir_para_out = Ir_para_out*R_para;

            It_perp_out = imtranslate(It_perp_out, [ 2*d*tand(theta_out), 0]);
            It_para_out = imtranslate(It_para_out, [ 2*d*tand(theta_out), 0]);
            I_perp_cell{i+1} = It_perp_out;
            I_para_cell{i+1} = It_para_out;
            
            It_perp_out = Ir_perp_out;
            It_para_out = Ir_para_out;
        end
    end
    % process images
    I_perp_out = I_perp_cell{1};
    I_para_out = I_para_cell{1};
    for i = 2:length(I_perp_cell),
        I_perp_out = I_perp_out + I_perp_cell{i};
        I_para_out = I_para_out + I_para_cell{i};
    end
    
end

