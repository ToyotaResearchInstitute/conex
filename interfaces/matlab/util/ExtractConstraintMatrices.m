function constraints = ExtractConstraintMatrices(A, affine, K, conex_format);

if nargin < 4
  conex_format = 0;
end

[r, c, v] = find(A);

s = 1;
col_start = 1;
for i = 1:length(K.s)
  col_end = col_start + K.s(i) * K.s(i);
  e = find(c(s:end) < col_end, 1, 'last'); e = e + s - 1;
  constraints{i}.matrix_row = r(s:e);
  constraints{i}.matrix_col = c(s:e) - col_start + 1;
  constraints{i}.matrix_val = v(s:e);
  s = e + 1;

  col_start = col_end; 
end

s = 1;
affine = full(affine);
for i = 1:length(K.s)
  e = s + K.s(i)  * K.s(i) - 1;
  constraints{i}.affine = affine(s:e);
  s = e + 1;
end

for i = 1:length(K.s)
  %Example: 'assignment' for input (2, 5, 5, 9) is (1, 2, 2, 3)
  [rows_unique, ~, assignment] = unique(constraints{i}.matrix_row);
  constraints{i}.matrix_row_assign = assignment;
  constraints{i}.variables = rows_unique;
  constraints{i}.num_cols = K.s(i) * K.s(i);
  constraints{i}.num_rows = length(rows_unique);
end

if conex_format
  for i = 1:length(K.s)
    n = K.s(i);
    constraints{i}.matrix_conex_format = ...
            full(sparse(constraints{i}.matrix_row_assign, constraints{i}.matrix_col,  ...
                 constraints{i}.matrix_val, constraints{i}.num_rows, constraints{i}.num_cols))';
    constraints{i}.matrix_conex_format = reshape(constraints{i}.matrix_conex_format, n, constraints{i}.num_rows*n);
  end
end

