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

auto get_piece_distrib(std::ifstream &stream) {
    std::array<double, 24> result{0};
    Game game;
    size_t total_count = 0;
    while (stream >> game) {
        for (auto pos: game) {
            auto piece_count = Bits::pop_count(pos.BP | pos.WP);
            result[piece_count] += 1;
            total_count += 1;
        }
    }
    for (auto &res: result) {
        res /= static_cast<double>(total_count);
    }
    return result;
}

auto get_piece_distrib(std::string input) {
    std::ifstream stream(input, std::ios::binary);
    return get_piece_distrib(stream);
}

auto get_capture_distrib(std::ifstream& stream){
    std::map<size_t,size_t> captures;

    std::istream_iterator<Game>begin(stream);
    std::istream_iterator<Game>end;

    std::for_each(begin,end,[&](Game game){
        for(auto pos : game){
            MoveListe liste;
            get_moves(pos,liste);
            for(Move m : liste){
                auto count = Bits::pop_count(m.captures);
                if(captures.count(count)>0){
                    captures[count]+=1;
                }else{
                    captures[count]=1;
                }
            }


        }
    });
    return captures;
}

 auto get_capture_distrib(std::string input){
    std::ifstream stream(input);
    return get_capture_distrib(stream);
}

size_t count_trainable_positions(std::string game_file, std::pair<size_t, size_t> range)
{
    std::ifstream stream(game_file, std::ios::binary);
    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game> end;
    size_t counter{0};
    // temporary before I can speed this thing up
    // way too slow
    std::for_each(begin, end, [&](const Game &g)
                  { counter += g.indices.size() + 1; });
    return counter;
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