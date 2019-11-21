%Supervised Laplacian Method
function [ correct_classified, wrong_classified ] = FNC_TRAIN_TEST_CLASSICAL_EMBEDDING_METHOD( train_data,test_data,nu_of_classes,dimension_embedded,Knn,beta,sigma,gamma )


pdist_training = squareform(pdist(double(train_data'))); % pairwise distance matrix of whole training data

[~,numb_of_training] = size(train_data);
[~,numb_of_test] = size(test_data);

%
numb_of_test_perclass = numb_of_test/nu_of_classes;
N = numb_of_training/nu_of_classes;
Kw = Knn;
Kb = Knn;

weights_list = cell(nu_of_classes,nu_of_classes);
for iclass=1:nu_of_classes
    train_data_for_one_class = train_data(:,(iclass-1)*N+1:iclass*N);
    pdist_mat = squareform(pdist(double(train_data_for_one_class')));
    
    weight_mat = exp(-(pdist_mat.^2)/beta);
    [valrow, indrow] = sort(pdist_mat,2); %sorts the elements in each row
    
    WA = zeros(N,N);
    for i=1:N
        WA(i,indrow(i,2:Kw+1)) = weight_mat(i,indrow(i,2:Kw+1));
    end
    WA = (WA+WA')/2;
    
    weights_list{iclass,iclass} = WA;
end

for iclass=1:nu_of_classes
    for jclass=iclass+1:nu_of_classes
        
        train_data_A = train_data(:,(iclass-1)*N+1:iclass*N);
        train_data_B = train_data(:,(jclass-1)*N+1:jclass*N);
        
        pdist_mat1 = squareform(pdist(double([train_data_A train_data_B]')));
        pdist_mat2 = pdist_mat1((1:N),(N+1:2*N));
        
        weightAB = exp(-(pdist_mat2.^2)/beta);
        [valrow, indrow] = sort(pdist_mat2,2);
        
        WAB = zeros(N,N);
        for i=1:N
            WAB(i,indrow(i,1:Kb)) = weightAB(i,indrow(i,1:Kb));
        end
        
        distBA = pdist_mat2';
        weightBA = weightAB';
        [valrow, indrow] = sort(distBA,2);
        
        WBA = zeros(N,N);
        for i=1:N
            WBA(i,indrow(i,1:Kb)) = weightBA(i,indrow(i,1:Kb));
        end
        
        weights_list{iclass,jclass}=(WAB+WBA')/2;
        weights_list{jclass,iclass}=(WAB'+WBA)/2;
        
    end
end


num_img_classes = zeros(1,nu_of_classes);
for g=1:nu_of_classes
    num_img_classes(g) = N;
end
imind = [0 sum(triu(toeplitz(num_img_classes)),1)];

W = zeros(numb_of_training,numb_of_training);
for iclass=1:nu_of_classes
    for jclass=1:nu_of_classes
        if(~isempty(weights_list{iclass,jclass}))
            W( imind(iclass)+1: imind(iclass+1), imind(jclass)+1: imind(jclass+1) )=weights_list{iclass,jclass};
        end
    end
end

D = diag(sum(W,1));
sqrt_invD = sqrt(inv(D));
L = sqrt_invD*(D-W)*sqrt_invD;

[V, eigvalL] = eig(L);
eigvalL = diag(eigvalL);
[eigsort, indsort]=sort(eigvalL);
V=V(:,indsort);
eigvalL=eigvalL(indsort);
min_nonzero_eigind=  find(eigvalL>1e-10, 1 );
eigvalL=diag(eigvalL);


% Class-discriminative weight matrix
WM_wit=zeros(numb_of_training,numb_of_training);
for iclass=1:nu_of_classes
    WM_wit(imind(iclass)+1: imind(iclass+1), imind(iclass)+1: imind(iclass+1))=1;
end

% Intra-class component
W_wit=W.*WM_wit;
D_wit=sum(W_wit,1);

inv_D_wit=D_wit.^(-1);
inv_D_wit=diag(inv_D_wit);
D_wit=diag(D_wit);
L_wit=sqrt(inv_D_wit)*(D_wit - W_wit)*sqrt(inv_D_wit);


% Inter-class component
W_bet=W-W_wit;
D_bet=sum(W_bet,1);
Vol_bet=sum(D_bet)/2;
pos_ind_D_bet=find(D_bet>0);
inv_D_bet=zeros(numb_of_training,1);
inv_D_bet(pos_ind_D_bet)=D_bet(pos_ind_D_bet).^(-1);
inv_D_bet=diag(inv_D_bet);
D_bet=diag(D_bet);
% Normalize with the between-class degree matrix
L_bet=sqrt(inv_D_bet)*(D_bet - W_bet)*sqrt(inv_D_bet);

L= L_wit- gamma*L_bet;

[V, eigvalL]=eig(L);
eigvalL=diag(eigvalL);
[eigsort, indsort]=sort(eigvalL);
V=V(:,indsort);

data_SL_classical = V(:,1:dimension_embedded);


% Applying Test Samples
ksi_matrix = KSI(pdist_training,sigma); % KSI
ksi_matrix_inv = inv(ksi_matrix);
fi_classical = ksi_matrix_inv * data_SL_classical;

[~, sz2] = size(test_data); % Test Datalar? RBF'e Uygulan齳or
new_points1 = [];
for idx = 1:sz2
    pdist_training_and_new_data = squareform(pdist(double([train_data';(test_data(:,idx))'])));
    [sz1,sz2] = size(pdist_training_and_new_data);
    dist_vect_new_pnt = pdist_training_and_new_data(sz1,(1:sz2-1));
    KSI_vect_new_pnt = KSI(dist_vect_new_pnt,sigma); % KSI
    new_points_classical_method(idx,:) = KSI_vect_new_pnt*fi_classical;
end

correct_classified = 0;
wrong_classified = 0;
for cntr = 1:nu_of_classes
    for i1 = (cntr-1)*numb_of_test_perclass+1:cntr*numb_of_test_perclass
        pdist_trainmapped_newtestsamplemapped = squareform(pdist([data_SL_classical;new_points_classical_method(i1,:)]));
        pdist_trainmapped_newtestsamplemapped_vector = pdist_trainmapped_newtestsamplemapped(end,(1:end-1));
        [nearest_dist,nearest_dist_ind] = min(pdist_trainmapped_newtestsamplemapped_vector);

        if nearest_dist_ind <= cntr*N && nearest_dist_ind >= (cntr-1)*N+1
            correct_classified = correct_classified + 1;
        else
            wrong_classified = wrong_classified + 1;
        end
    end
end


end

