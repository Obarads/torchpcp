#ifndef _BALL_QUERY_HPP
#define _BALL_QUERY_HPP

#include <torch/extension.h>
#include <vector>

at::Tensor ball_query_forward(at::Tensor centers_coords,
                              at::Tensor points_coords, 
                              const float radius,
                              const int num_neighbors);

// std::vector<at::Tensor> ball_query_forward(at::Tensor centers_coords,
//                               at::Tensor points_coords, 
//                               const float radius,
//                               const int num_neighbors);


#endif
