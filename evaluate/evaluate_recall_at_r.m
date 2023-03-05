load results
load B_for_base
load sift1M

all_recall = zeros(10000, 1);

for q_idx = 1:10000
    
    q_idx
    
    q = query(:,q_idx);

    look_up = zeros(2, 256);
    for ii = 1:2
        for jj = 1:256
            this_word = C(ii,:,jj)';
            diff = q - this_word;
            look_up(ii,jj) = diff' * diff;
        end
    end

    dist = zeros(1000000, 1);
    for ii = 1:1000000
        idx_1 = B_for_base(ii, 1) + 1;
        idx_2 = B_for_base(ii, 2) + 1;
        dist(ii,1) = look_up(1, idx_1) + look_up(2, idx_2);
    end
    [a,b] = sort(dist);
    b_top50 = b(1:50,1);
    c = intersect(b_top50 - 1, groundtruth(1:50, q_idx) );
    recall = size(c, 1) / 50;
    all_recall(q_idx,1) = recall;
end



%{
dist_es = zeros(1000000, 1);
q = query(:,1);
q_rep = repmat(q, 1, 1000000);
diff_es = q_rep - base;
for ii = 1:1000000
    temp = diff_es(:,ii);
    dist_es(ii,1) = temp' * temp;
end

[a_es, b_es] = sort(dist_es);
b_es_top50 = b_es(1:50, 1);
c_es = intersect(b_es_top50 - 1, groundtruth(1:50, 1) );
recall_es = size(c_es, 1) / 50;
%}
