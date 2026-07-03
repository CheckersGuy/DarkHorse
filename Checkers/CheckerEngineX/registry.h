#ifndef REGISTRY
#define REGISTRY

#include "windows.h"
#include <iostream>
#include <optional>
#include <string>
namespace Registry {

constexpr inline int REG_MSG_SIZE = 100;

void reg_set_int(const char *name, int val);
int reg_get_int(const char *name, int *val);

void reg_set_string(const char *name, const char *string);
int reg_get_string(const char *name, char *string, int maxlen);

void set_db_size(int val);
void set_hash_size(int val);
void set_db_path(std::string path);
void set_enable_wld(bool enable);
void set_max_db_pieces(int val);
std::optional<int> get_db_size();
std::optional<int> get_hash_size();
std::optional<std::string> get_db_path();
std::optional<int> get_max_db_pieces();
bool use_wld_db();

} // namespace Registry
#endif
