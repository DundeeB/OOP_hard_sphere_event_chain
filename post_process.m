function [] = post_process(output_file_name, lib, varargin)
display(['Post processing for library: ' lib]);
addpath('../post_process/');
% [S45, Sm45, Sm_theta, k45, theta] = Bragg_post_process(varargin{2:end});
[psi16, psi14, psi23] = psi_post_process_for_lib(lib, varargin{:});
[b3, M3, N_sp3] = M_frustration_post_proccess_for_lib(3, lib, varargin{:});

% save([lib '\' output_file_name],...
%     'psi16','psi14','psi23', 'b', 'M','N_sp',...
%     'S45', 'Sm45', 'Sm_theta', 'k45', 'theta');
save([lib '\' output_file_name],...
    'psi16','psi14','psi23', 'b3','M3','N_sp3');
end