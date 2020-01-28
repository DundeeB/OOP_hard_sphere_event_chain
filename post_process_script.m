%% specific libraries
% lib='../simulation-results/N=900_h=1_rhoH=0.8365018104736054_AF_square_ECMC/';
% post_process('post_proccess100',lib,100,true);
% 
% lib='../simulation-results/N=900_h=1_rhoH=0.8365018104736054_AF_triangle_ECMC/';
% post_process('post_proccess100',lib,100,true);

% lib = '../simulation-results/N=3600_h=1_rhoH=0.8559553409497356_AF_triangle_ECMC/';
% post_process('post_proccess10',lib,10,true);

%% all of interest

father_dir = 'C:\Users\Daniel\OneDrive - Technion\simulation-results\';
folds_obj = dir(father_dir);
sim_dirs = {};
for i=1:length(folds_obj)
    f = folds_obj(i).name;
    if sum(strcmp(f,{'.','..','Small or 2D simulations'}))||...
            ~isdir([father_dir f])  
        continue
    end
    sim_dirs{end+1} = [father_dir f];
end
n = length(sim_dirs);
rho_H_vec = zeros(n,1);
h_vec = rho_H_vec;
N_vec = rho_H_vec;
ic = {};
for i=1:n
    N_vec(i) = str2double(regexprep(regexprep(...
        sim_dirs{i},'_h=.*',''),'.*N=',''));
    h_vec(i) = str2double(regexprep(regexprep(...
        sim_dirs{i},'.*h=',''),'_rhoH.*',''));
    rho_H_vec(i) = str2double(regexprep(regexprep(...
        sim_dirs{i},'.*rhoH=',''),'_.*',''));
    ic{i} = regexprep(sim_dirs{i},'.*rhoH=[0-9]*\.[0-9]*_','');

    if regexpi(ic{i},'ECMC') & ~isnan(h_vec(i))
        post_process('output_psi_frustration100',sim_dirs{i},100,true);
    end
end