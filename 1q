#include "Compress.h"


std::pair<size_t, size_t> count_unique_positions(std::string game_file) {
    std::ifstream stream(game_file, std::ios::binary);
    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game> end;
    std::vector<Game> games;
    std::copy(begin,end,std::back_inserter(games));
    return count_unique_positions(games.begin(), games.end());
}

std::optional<std::string> is_temporary_train_file(std::string name) {
    std::regex reg("[a-z0-9\\-\\_]+[.]train[.]temp[0-9]+");
    if (std::regex_match(name, reg)) {
        auto f = name.find('.');
        return std::make_optional(name.substr(0, f));
    }
    return std::nullopt;
}

void create_subset(std::string file,std::string output,size_t num_games){
    std::ifstream stream(file,std::ios::binary);
    std::ofstream out_stream(output,std::ios::binary);
    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game> end;
    std::copy_n(begin,num_games,std::ostream_iterator<Game>(out_stream));
}


void merge_temporary_files(std::string directory, std::string out_directory)
{
    std::filesystem::path in_path(directory);
    std::filesystem::path out_path(out_directory);
    if (!std::filesystem::is_directory(in_path))
    {
        throw std::string{"Not a directory as in_path"};
    }
    if (!std::filesystem::is_directory(out_path))
    {
        throw std::string{"Not a directory as out_path"};
    }
    // finding all relevant files
    std::map<std::string, std::vector<std::filesystem::path>> my_map;
    for (auto &ent : std::filesystem::directory_iterator(in_path))
    {
        auto path = ent.path();
        auto t = is_temporary_train_file(path.filename().string());
        if (t.has_value())
        {
            my_map[t.value()].emplace_back(path);
        }
    }
    // listing all the temporary files found
    for (auto &val : my_map)
    {
        std::cout << val.first << std::endl;
        std::vector<std::filesystem::path> local_paths;

        for (auto &path_file : val.second)
        {
            local_paths.emplace_back(path_file.c_str());
        }
        merge_training_data(local_paths.begin(), local_paths.end(), out_directory + val.first + ".train");

        for (auto &path_file : val.second)
        {
            std::filesystem::remove_all(path_file);
        }
    }
}
