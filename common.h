double* alloc_matrix(int size);
void init_matrix(double* matrix, int size);
void fill_matrix(double* matrix, int size, int fill_value);
void print_matrix(double* matrix, int size);
void block_multiply(double* a_block, double* b_block,
                    double* c_block, int block_size);