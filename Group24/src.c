#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include<string.h>

//Reading file using assigned leaders
float* read_input_file_parallel_optimized(const char* filename, int NX, int NY, int NZ, int NC,
    int local_nx, int local_ny, int local_nz, int rank, int PX, int PY, int PZ) {
    MPI_Status status;
    MPI_File fh;
    MPI_Info info;
    
    // Calculate process coordinates
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);
    
    // Define aggregator ranks - one per Z-layer
    int z_group_size = PX * PY;
    int aggregator_rank = pz * z_group_size; // First process in each Z-layer is the aggregator
    bool is_aggregator = (rank % z_group_size == 0);
    
    // Calculate local domain size
    int local_size = local_nx * local_ny * local_nz * NC;
    float* local_data = (float*)malloc(local_size * sizeof(float));
    if (local_data == NULL) {
        fprintf(stderr, "Memory allocation failed for local data array on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return NULL;
    }
    
    // Aggregators read the file
    if (is_aggregator) {
        // Create info object for I/O optimizations
        MPI_Info_create(&info);
        MPI_Info_set(info, "access_style", "read_once");
        MPI_Info_set(info, "collective_buffering", "true"); //enabling collective buffering for better performance
        MPI_Info_set(info, "cb_buffer_size", "16777216"); // 16MB buffer for each process to buffer
        
        // Open the file
        int ret = MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_RDONLY, info, &fh);
        if (ret != MPI_SUCCESS) {
            char error_string[MPI_MAX_ERROR_STRING];
            int length_of_error_string;
            MPI_Error_string(ret, error_string, &length_of_error_string);
            fprintf(stderr, "Error opening file %s: %s\n", filename, error_string);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return NULL;
        }
        MPI_Offset file_size;
        MPI_File_get_size(fh, &file_size);
        // Expected file size in bytes (NX*NY*NZ*NC floats)
        MPI_Offset expected_size = (MPI_Offset)(NX) * (MPI_Offset)(NY) * (MPI_Offset)(NZ) * (MPI_Offset)(NC) * sizeof(float);
        
        // Check if file size matches expected size
        if (file_size != expected_size) {
            if (rank == 0) {
                fprintf(stderr, "Error: File size mismatch. Expected %lld bytes (%d*%d*%d*%d floats), but got %lld bytes.\n", 
                        (long long)expected_size, NX, NY, NZ, NC, (long long)file_size);
            }
            MPI_File_close(&fh);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return NULL;
        }
        // Calculate the Z-layer offset for this aggregator
        int layer_start_z = pz * local_nz;
        
        // Allocate buffer for the entire Z-layer (all X,Y points for this Z range)
        size_t layer_data_size = (size_t)NX * NY * local_nz * NC;
        float* layer_data = (float*)malloc(layer_data_size * sizeof(float));
        if (layer_data == NULL) {
            fprintf(stderr, "Memory allocation failed for layer data on aggregator %d\n", rank);
            MPI_File_close(&fh);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return NULL;
        }
        
        // Create a datatype for reading the Z-layer
        MPI_Datatype time_block_type;
        MPI_Type_contiguous(NC, MPI_FLOAT, &time_block_type);
        MPI_Type_commit(&time_block_type);
        
        int ndims = 3;
        int array_of_sizes[3] = {NZ, NY, NX};
        int array_of_subsizes[3] = {local_nz, NY, NX};
        int array_of_starts[3] = {layer_start_z, 0, 0};
        
        MPI_Datatype layer_type;
        MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, 
                               array_of_starts, MPI_ORDER_C, time_block_type, &layer_type);
        MPI_Type_commit(&layer_type);
        
        // Set file view and read the data
        MPI_File_set_view(fh, 0, MPI_FLOAT, layer_type, "native", info);
        MPI_File_read_all(fh, layer_data, layer_data_size, MPI_FLOAT, &status);
        
        // Create a new datatype for the subdomain extraction
        MPI_Datatype subdomain_type;
        
        // Extract the aggregator's own portion first
        for (int z = 0; z < local_nz; z++) {
            for (int y = 0; y < local_ny; y++) {
                for (int x = 0; x < local_nx; x++) {
                    for (int c = 0; c < NC; c++) {
                        int local_idx = ((z * local_ny + y) * local_nx + x) * NC + c;
                        int layer_idx = ((z * NY + y) * NX + x) * NC + c;
                        local_data[local_idx] = layer_data[layer_idx];
                    }
                }
            }
        }
        
        // Create derived datatypes for sending subdomains to other processes from the aggregator
        MPI_Request requests[z_group_size - 1];
        MPI_Status statuses[z_group_size - 1];
        int req_idx = 0;
        
        // For each process in this Z-layer group we send them their subdomain using derived datatypes
        for (int p_rank = 0; p_rank < z_group_size; p_rank++) {
            // Skip self
            if (p_rank == 0) continue;
            
            int target_rank = pz * z_group_size + p_rank;
            int target_px = p_rank % PX;
            int target_py = p_rank / PX;
            
            // Calculate starting indices for this target process
            int target_start_x = target_px * local_nx;
            int target_start_y = target_py * local_ny;
            
            // Create source datatype (subdomain within layer data)
            int src_sizes[3] = {local_nz, NY, NX};
            int src_subsizes[3] = {local_nz, local_ny, local_nx};
            int src_starts[3] = {0, target_start_y, target_start_x};
            
            MPI_Datatype src_type;
            MPI_Type_create_subarray(3, src_sizes, src_subsizes, src_starts,
                                   MPI_ORDER_C, time_block_type, &src_type);
            MPI_Type_commit(&src_type);
            
            // Send the data using the derived datatype
            MPI_Isend(layer_data, 1, src_type, target_rank, 0, MPI_COMM_WORLD, &requests[req_idx++]);
            
            // Clean up datatype after sending
            MPI_Type_free(&src_type);
        }
        
        // Wait for all sends to complete
        MPI_Waitall(req_idx, requests, statuses);
        
        // Clean up
        free(layer_data);
        MPI_Type_free(&layer_type);
        MPI_Type_free(&time_block_type);
        MPI_File_close(&fh);
        MPI_Info_free(&info);
    } 
    else {
        // Non-aggregator processes receive their data from the aggregator
        MPI_Recv(local_data, local_size, MPI_FLOAT, aggregator_rank, 0, MPI_COMM_WORLD, &status);
    }
    
    return local_data;
}

// Define the domain structure with pointers to local data and ghost regions for saving space
typedef struct {
    float *local_data;      // Pointer to the local domain data
    float *left_ghost;      // Pointer to left ghost plane (YZ plane at x=-1)
    float *right_ghost;     // Pointer to right ghost plane (YZ plane at x=local_nx)
    float *bottom_ghost;    // Pointer to bottom ghost plane (XZ plane at y=-1)
    float *top_ghost;       // Pointer to top ghost plane (XZ plane at y=local_ny)
    float *back_ghost;      // Pointer to back ghost plane (XY plane at z=-1)
    float *front_ghost;     // Pointer to front ghost plane (XY plane at z=local_nz)
    int local_nx, local_ny, local_nz, NC;  // Local domain dimensions
} DomainData;

// Function to exchange the boundaries and construct the Domain structure defined for each process
DomainData* communicate_boundaries_optimized(float *local_data, int local_nx, int local_ny, int local_nz, 
                                  int NC, int rank, int PX, int PY, int PZ) {
    MPI_Comm comm = MPI_COMM_WORLD;

    // Calculate process coordinates from rank
    int px = rank % PX;
    int py = (rank / PX) % PY;
    int pz = rank / (PX * PY);

    // Calculate neighbor ranks
    int left   = (px > 0) ? rank - 1 : MPI_PROC_NULL;
    int right  = (px < PX - 1) ? rank + 1 : MPI_PROC_NULL;
    int bottom = (py > 0) ? rank - PX : MPI_PROC_NULL;
    int top    = (py < PY - 1) ? rank + PX : MPI_PROC_NULL;
    int back   = (pz > 0) ? rank - (PX * PY) : MPI_PROC_NULL;
    int front  = (pz < PZ - 1) ? rank + (PX * PY) : MPI_PROC_NULL;

    // Allocate the domain structure
    DomainData *domain = (DomainData*)malloc(sizeof(DomainData));
    if (!domain) {
        fprintf(stderr, "Failed to allocate domain structure on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return NULL;
    }

    // Store local domain dimensions and data pointer
    domain->local_nx = local_nx;
    domain->local_ny = local_ny;
    domain->local_nz = local_nz;
    domain->NC = NC;
    domain->local_data = local_data;

    // Allocate memory for ghost regions
    size_t yz_plane_size = local_ny * local_nz * NC;
    size_t xz_plane_size = local_nx * local_nz * NC;
    size_t xy_plane_size = local_nx * local_ny * NC;

    domain->left_ghost = (float*)malloc(yz_plane_size * sizeof(float));
    domain->right_ghost = (float*)malloc(yz_plane_size * sizeof(float));
    domain->bottom_ghost = (float*)malloc(xz_plane_size * sizeof(float));
    domain->top_ghost = (float*)malloc(xz_plane_size * sizeof(float));
    domain->back_ghost = (float*)malloc(xy_plane_size * sizeof(float));
    domain->front_ghost = (float*)malloc(xy_plane_size * sizeof(float));

    // Initialize ghost regions with FLT_MAX
    for (int i = 0; i < yz_plane_size; i++) {
        domain->left_ghost[i] = FLT_MAX;
        domain->right_ghost[i] = FLT_MAX;
    }
    for (int i = 0; i < xz_plane_size; i++) {
        domain->bottom_ghost[i] = FLT_MAX;
        domain->top_ghost[i] = FLT_MAX;
    }
    for (int i = 0; i < xy_plane_size; i++) {
        domain->back_ghost[i] = FLT_MAX;
        domain->front_ghost[i] = FLT_MAX;
    }

    // Create a datatype for a time block
    MPI_Datatype time_block_type;
    MPI_Type_contiguous(NC, MPI_FLOAT, &time_block_type);
    MPI_Type_commit(&time_block_type);

    // Create datatypes for the interior faces of our local domain using the time_block_type

    // YZ plane at x=0 (left face)
    MPI_Datatype yz_face_send_left;
    {
        int ndims = 3;
        int array_of_sizes[3] = {local_nz, local_ny, local_nx};
        int array_of_subsizes[3] = {local_nz, local_ny, 1};
        int array_of_starts[3] = {0, 0, 0};
        MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, 
                                array_of_starts, MPI_ORDER_C, time_block_type, &yz_face_send_left);
        MPI_Type_commit(&yz_face_send_left);
    }

    // YZ plane at x=local_nx-1 (right face)
    MPI_Datatype yz_face_send_right;
    {
        int ndims = 3;
        int array_of_sizes[3] = {local_nz, local_ny, local_nx};
        int array_of_subsizes[3] = {local_nz, local_ny, 1};
        int array_of_starts[3] = {0, 0, local_nx-1};
        MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, 
                                array_of_starts, MPI_ORDER_C, time_block_type, &yz_face_send_right);
        MPI_Type_commit(&yz_face_send_right);
    }

    // XZ plane at y=0 (bottom face)
    MPI_Datatype xz_face_send_bottom;
    {
        int ndims = 3;
        int array_of_sizes[3] = {local_nz, local_ny, local_nx};
        int array_of_subsizes[3] = {local_nz, 1, local_nx};
        int array_of_starts[3] = {0, 0, 0};
        MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, 
                                array_of_starts, MPI_ORDER_C, time_block_type, &xz_face_send_bottom);
        MPI_Type_commit(&xz_face_send_bottom);
    }

    // XZ plane at y=local_ny-1 (top face)
    MPI_Datatype xz_face_send_top;
    {
        int ndims = 3;
        int array_of_sizes[3] = {local_nz, local_ny, local_nx};
        int array_of_subsizes[3] = {local_nz, 1, local_nx};
        int array_of_starts[3] = {0, local_ny-1, 0};
        MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, 
                                array_of_starts, MPI_ORDER_C, time_block_type, &xz_face_send_top);
        MPI_Type_commit(&xz_face_send_top);
    }

    // XY plane at z=0 (back face)
    MPI_Datatype xy_face_send_back;
    {
        int ndims = 3;
        int array_of_sizes[3] = {local_nz, local_ny, local_nx};
        int array_of_subsizes[3] = {1, local_ny, local_nx};
        int array_of_starts[3] = {0, 0, 0};
        MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, 
                                array_of_starts, MPI_ORDER_C, time_block_type, &xy_face_send_back);
        MPI_Type_commit(&xy_face_send_back);
    }

    // XY plane at z=local_nz-1 (front face)
    MPI_Datatype xy_face_send_front;
    {
        int ndims = 3;
        int array_of_sizes[3] = {local_nz, local_ny, local_nx};
        int array_of_subsizes[3] = {1, local_ny, local_nx};
        int array_of_starts[3] = {local_nz-1, 0, 0};
        MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, 
                                array_of_starts, MPI_ORDER_C, time_block_type, &xy_face_send_front);
        MPI_Type_commit(&xy_face_send_front);
    }

    // Exchange data with neighbors using non-blocking communication
    MPI_Request requests[12];
    MPI_Status statuses[12];
    int req_count = 0;

    // Exchange in X direction
    MPI_Isend(local_data, 1, yz_face_send_left, left, 0, comm, &requests[req_count++]);
    MPI_Irecv(domain->right_ghost, yz_plane_size, MPI_FLOAT, right, 0, comm, &requests[req_count++]);

    MPI_Isend(local_data, 1, yz_face_send_right, right, 1, comm, &requests[req_count++]);
    MPI_Irecv(domain->left_ghost, yz_plane_size, MPI_FLOAT, left, 1, comm, &requests[req_count++]);

    // Exchange in Y direction
    MPI_Isend(local_data, 1, xz_face_send_bottom, bottom, 2, comm, &requests[req_count++]);
    MPI_Irecv(domain->top_ghost, xz_plane_size, MPI_FLOAT, top, 2, comm, &requests[req_count++]);

    MPI_Isend(local_data, 1, xz_face_send_top, top, 3, comm, &requests[req_count++]);
    MPI_Irecv(domain->bottom_ghost, xz_plane_size, MPI_FLOAT, bottom, 3, comm, &requests[req_count++]);

    // Exchange in Z direction
    MPI_Isend(local_data, 1, xy_face_send_back, back, 4, comm, &requests[req_count++]);
    MPI_Irecv(domain->front_ghost, xy_plane_size, MPI_FLOAT, front, 4, comm, &requests[req_count++]);

    MPI_Isend(local_data, 1, xy_face_send_front, front, 5, comm, &requests[req_count++]);
    MPI_Irecv(domain->back_ghost, xy_plane_size, MPI_FLOAT, back, 5, comm, &requests[req_count++]);

    // Wait for all communications to complete
    MPI_Waitall(req_count, requests, statuses);

    // Free the datatypes
    MPI_Type_free(&time_block_type);
    MPI_Type_free(&yz_face_send_left);
    MPI_Type_free(&yz_face_send_right);
    MPI_Type_free(&xz_face_send_bottom);
    MPI_Type_free(&xz_face_send_top);
    MPI_Type_free(&xy_face_send_back);
    MPI_Type_free(&xy_face_send_front);

    return domain;
}

// Function to find local minima and maxima using the optimized structure
void find_local_extrema_optimized(DomainData *domain, int *local_min_count, int *local_max_count, 
                                 float *global_min, float *global_max) {
    
    int local_nx = domain->local_nx;
    int local_ny = domain->local_ny;
    int local_nz = domain->local_nz;
    int NC = domain->NC;
    float *local_data = domain->local_data;
    
    // Initialize the arrays for each time step
    for (int t = 0; t < NC; t++) {
        local_min_count[t] = 0;
        local_max_count[t] = 0;
        global_min[t] = FLT_MAX;
        global_max[t] = -FLT_MAX;
    }
    
    // Check each interior point for local extrema
    for (int t = 0; t < NC; t++) {
        for (int z = 0; z < local_nz; z++) {
            for (int y = 0; y < local_ny; y++) {
                for (int x = 0; x < local_nx; x++) {
                    int local_idx = ((z * local_ny + y) * local_nx + x) * NC + t;
                    float val = local_data[local_idx];
                    
                    // Update global min and max
                    if (val < global_min[t]) global_min[t] = val;
                    if (val > global_max[t]) global_max[t] = val;
                    
                    // Get values from neighbors or ghost regions as needed
                    float left_val, right_val, bottom_val, top_val, back_val, front_val;
                    
                    // Get left neighbor value (either from local domain or left ghost)
                    if (x > 0) {
                        left_val = local_data[((z * local_ny + y) * local_nx + (x-1)) * NC + t];
                    } else {
                        left_val = domain->left_ghost[(z * local_ny + y) * NC + t];
                    }
                    
                    // Get right neighbor value (either from local domain or right ghost)
                    if (x < local_nx - 1) {
                        right_val = local_data[((z * local_ny + y) * local_nx + (x+1)) * NC + t];
                    } else {
                        right_val = domain->right_ghost[(z * local_ny + y) * NC + t];
                    }
                    
                    // Get bottom neighbor value (either from local domain or bottom ghost)
                    if (y > 0) {
                        bottom_val = local_data[((z * local_ny + (y-1)) * local_nx + x) * NC + t];
                    } else {
                        bottom_val = domain->bottom_ghost[(z * local_nx + x) * NC + t];
                    }
                    
                    // Get top neighbor value (either from local domain or top ghost)
                    if (y < local_ny - 1) {
                        top_val = local_data[((z * local_ny + (y+1)) * local_nx + x) * NC + t];
                    } else {
                        top_val = domain->top_ghost[(z * local_nx + x) * NC + t];
                    }
                    
                    // Get back neighbor value (either from local domain or back ghost)
                    if (z > 0) {
                        back_val = local_data[(((z-1) * local_ny + y) * local_nx + x) * NC + t];
                    } else {
                        back_val = domain->back_ghost[(y * local_nx + x) * NC + t];
                    }
                    
                    // Get front neighbor value (either from local domain or front ghost)
                    if (z < local_nz - 1) {
                        front_val = local_data[(((z+1) * local_ny + y) * local_nx + x) * NC + t];
                    } else {
                        front_val = domain->front_ghost[(y * local_nx + x) * NC + t];
                    }
                    
                    // Handle sentinel values for ghost regions
                    float left_val_min = left_val;
                    float right_val_min = right_val;
                    float bottom_val_min = bottom_val;
                    float top_val_min = top_val;
                    float back_val_min = back_val;
                    float front_val_min = front_val;

                    float left_val_max = left_val;
                    float right_val_max = right_val;
                    float bottom_val_max = bottom_val;
                    float top_val_max = top_val;
                    float back_val_max = back_val;
                    float front_val_max = front_val;

                    // For maximum check: if neighbor is FLT_MAX (ghost cell), replace with -FLT_MAX
                    if (left_val_min == FLT_MAX) left_val_max = -FLT_MAX;
                    if (right_val_min == FLT_MAX) right_val_max = -FLT_MAX;
                    if (bottom_val_min == FLT_MAX) bottom_val_max = -FLT_MAX;
                    if (top_val_min == FLT_MAX) top_val_max = -FLT_MAX;
                    if (back_val_min == FLT_MAX) back_val_max = -FLT_MAX;
                    if (front_val_min == FLT_MAX) front_val_max = -FLT_MAX;
                    
                    // Check if it's a local minimum
                    if (val < left_val_min && val < right_val_min && val < bottom_val_min && 
                        val < top_val_min && val < back_val_min && val < front_val_min) {
                        local_min_count[t]++;
                    }
                    
                    // Check if it's a local maximum
                    if (val > left_val_max && val > right_val_max && val > bottom_val_max && 
                        val > top_val_max && val > back_val_max && val > front_val_max) {
                        local_max_count[t]++;
                    }
                }
            }
        }
    }
}

// Function to free the domain structure along with its contents
void free_domain_data(DomainData *domain) {
    if (domain) {
        // Don't free local_data as it's managed elsewhere
        free(domain->left_ghost);
        free(domain->right_ghost);
        free(domain->bottom_ghost);
        free(domain->top_ghost);
        free(domain->back_ghost);
        free(domain->front_ghost);
        free(domain);
    }
}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check if we have the correct number of arguments
    if (argc != 10) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input_file> <PX> <PY> <PZ> <NX> <NY> <NZ> <NC> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Parse command line arguments
    char* input_filename = argv[1];
    int PX = atoi(argv[2]);
    int PY = atoi(argv[3]);
    int PZ = atoi(argv[4]);
    int NX = atoi(argv[5]);
    int NY = atoi(argv[6]);
    int NZ = atoi(argv[7]);
    int NC = atoi(argv[8]);
    char* output_filename = argv[9];
    
    // Verify that PX * PY * PZ equals size and other constraints
    if (PX * PY * PZ != size|| PX < 1 || PY < 1 || PZ < 1 || NX<=0 || NY<=0 || NZ<=0 ||
    NX > 1024 || NY > 1024 || NZ > 1024 || NC > 1000 ||
    NX % PX != 0 || NY % PY != 0 || NZ % PZ != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Invalid parameters\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Start timing
    double t1 = MPI_Wtime();

    // Calculate local dimensions
    int local_nx = NX / PX;
    int local_ny = NY / PY;
    int local_nz = NZ / PZ;
    int local_size = local_nx * local_ny * local_nz * NC;

    //Obtain the data for that process
    float* subdom_data = read_input_file_parallel_optimized(input_filename, NX, NY, NZ, NC, 
    local_nx, local_ny, local_nz, rank, PX, PY, PZ);
    if (!subdom_data) {
        fprintf(stderr, "Error reading input file on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    double t2 = MPI_Wtime();
    // Exchange boundary planes to get ghost cells, create the extended domain structure
    DomainData* domain = communicate_boundaries_optimized(subdom_data, local_nx, local_ny, local_nz, NC, rank, PX, PY, PZ);
    
    // Allocate arrays for local results per time step
    int* local_min_count = (int*)malloc(NC * sizeof(int));
    int* local_max_count = (int*)malloc(NC * sizeof(int));
    float* local_global_min = (float*)malloc(NC * sizeof(float));
    float* local_global_max = (float*)malloc(NC * sizeof(float));
    
    // Find local extrema
    find_local_extrema_optimized(domain, local_min_count, local_max_count, local_global_min, local_global_max);
    
    // Allocate arrays for global results
    int* total_min_count = NULL;
    int* total_max_count = NULL;
    float* overall_min = NULL;
    float* overall_max = NULL;
    
    //allocate  only at rank 0 as significant only there
    if (rank == 0) {
        total_min_count = (int*)malloc(NC * sizeof(int));
        total_max_count = (int*)malloc(NC * sizeof(int));
        overall_min = (float*)malloc(NC * sizeof(float));
        overall_max = (float*)malloc(NC * sizeof(float));
    }
    
    // Reduce results across all processes
    MPI_Reduce(local_min_count, total_min_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max_count, total_max_count, NC, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_global_min, overall_min, NC, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_global_max, overall_max, NC, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Record time after computation and communication
    double t3 = MPI_Wtime();
    
    // Write results to output file (only rank 0)
    if (rank == 0) {
        FILE* outfile = fopen(output_filename, "w");
        if (outfile == NULL) {
            fprintf(stderr, "Error opening output file: %s\n", output_filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Write count of local minima and maxima for each time step
        for (int t = 0; t < NC; t++) {
            fprintf(outfile, "(%d, %d)", total_min_count[t], total_max_count[t]);
            if (t < NC - 1) {
                fprintf(outfile, ", ");
            }
        }
        fprintf(outfile, "\n");
        
        // Write global minimum and maximum for each time step
        for (int t = 0; t < NC; t++) {
            fprintf(outfile, "(%f, %f)", overall_min[t], overall_max[t]);
            if (t < NC - 1) {
                fprintf(outfile, ", ");
            }
        }
        fprintf(outfile, "\n");
        // Calculate maximum times across all processes
        double max_read_time, max_maincode_time, max_total_time;
        double read_time = t2 - t1;
        double maincode_time = t3 - t2;
        double total_time = t3 - t1;

        //reduce all times
        MPI_Reduce(&read_time, &max_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&maincode_time, &max_maincode_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        // Write timing information
        fprintf(outfile, "%f, %f, %f\n", max_read_time, max_maincode_time, max_total_time);
        
        fclose(outfile);
        
        // Free allocated memory for global results
        free(total_min_count);
        free(total_max_count);
        free(overall_min);
        free(overall_max);
    } 
    else {
        // Non-root processes still need to participate in the timing reduction
        double read_time = t2 - t1;
        double maincode_time = t3 - t2;
        double total_time = t3 - t1;
        MPI_Reduce(&read_time, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&maincode_time, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&total_time, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    
    // Free allocated memory
    free(subdom_data);
    free_domain_data(domain);
    free(local_min_count);
    free(local_max_count);
    free(local_global_min);
    free(local_global_max);
    MPI_Finalize();
    return 0;
}