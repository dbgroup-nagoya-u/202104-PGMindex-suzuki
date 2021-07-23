/*
 * Testing knn query.
 */

#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "pgm/pgm_index_variants.hpp"


std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> acc_knn(std::tuple<uint64_t, uint64_t, uint64_t> query_point, std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> &data , int k);
double calc_recall(std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>> &acc , std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>> &pred);

int main() {
    // Generate random points in a 3D space
    auto rand_coord = []() { return std::rand() % 500; };
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> data(1000000);
    std::generate(data.begin(), data.end(), [&] { return std::make_tuple(rand_coord(), rand_coord(), rand_coord()); });

    // Construct the Multidimensional PGM-index
    constexpr auto dimensions = 3; // number of dimensions
    constexpr auto epsilon = 32;   // space-time trade-off parameter
    pgm::MultidimensionalPGMIndex<dimensions, uint64_t, epsilon> pgm_3d(data.begin(), data.end());

    // Generate 100 query points
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> query(100);
    std::generate(query.begin(), query.end(), [&] { return std::make_tuple(rand_coord(), rand_coord(), rand_coord()); });

    // knn query for 100 times
    std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>> ans_knn;
    auto start = std::chrono::high_resolution_clock::now();
    for(auto point : query){
        auto ans = pgm_3d.knn(point , 25);
        ans_knn.push_back(ans);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() * 1.0 / query.size();
    std::cout << "knn query time , " << time << std::endl;

    // ACC knn query for 100 times
    std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>> acc_ans_knn;
    for(auto point : query){
        std::vector<std::tuple<uint64_t , uint64_t, uint64_t>> ans;
        ans = acc_knn(point , data , 25);
        acc_ans_knn.push_back(ans);
    }
    std::cout << "knn query recall , " << calc_recall(acc_ans_knn , ans_knn) << std::endl;

    return 0;
}

double calc_recall(std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>> &acc , std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>> &pred){
    int all = 0;
    double sum = 0.0;
    double product = 1.0;
    int all_acc = acc.size();
    for(int i = 0 ; i < acc.size() ; i++){
        all += acc[i].size();
        int acc_size = acc[i].size();
        int collect_size = 0;
        std::map<uint64_t, std::vector<std::tuple<uint64_t, uint64_t>>> ans;
        for(auto acc_point : acc[i]){   
            uint64_t ans_x = std::get<0>(acc_point);
            uint64_t ans_y = std::get<1>(acc_point);
            uint64_t ans_z = std::get<2>(acc_point);
            if(ans.count(ans_x) == 0) ans.insert(std::pair<uint64_t,std::vector<std::tuple<uint64_t, uint64_t>>> (ans_x, std::vector<std::tuple<uint64_t, uint64_t>>()));
            ans[ans_x].push_back(std::make_tuple(ans_y, ans_z));
        }
        for(auto pred_point : pred[i]){
            uint64_t pred_x = std::get<0>(pred_point);
            uint64_t pred_y = std::get<1>(pred_point);
            uint64_t pred_z = std::get<2>(pred_point);
            std::tuple<uint64_t, uint64_t> pred_yz{std::get<1>(pred_point), std::get<2>(pred_point)};
            if (std::find(ans[pred_x].begin() , ans[pred_x].end(), pred_yz) != ans[pred_x].end()) collect_size += 1;
        }
        if(acc_size == 0){
            all_acc -= 1;
            continue;
        }
        double accuracy = collect_size * 1.0 / acc_size;
        sum += accuracy;
        product *= accuracy;
    }
    return sum * 1.0 / all_acc;
}

std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> acc_knn(std::tuple<uint64_t , uint64_t, uint64_t> query_point, std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> &data , int k){
    auto dist = [&](std::tuple<uint64_t , uint64_t, uint64_t> point){
        return std::sqrt(std::pow((int64_t)std::get<0>(query_point) - (int64_t)std::get<0>(point), 2) + std::pow((int64_t)std::get<1>(query_point) - (int64_t)std::get<1>(point), 2) + std::pow((int64_t)std::get<2>(query_point) - (int64_t)std::get<2>(point), 2));
    };
    std::sort(data.begin() , data.end() , [&](auto const& lhs, auto const& rhs) {
        double dist_l = dist(lhs);
        double dist_r = dist(rhs);
        return dist_l < dist_r;
    });
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> ans{data.begin(), data.begin() + k};
    return ans;
}