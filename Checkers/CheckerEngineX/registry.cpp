
#include "registry.h"
#define GC_REGISTRY_NAME "Software\\DarkHorse"

namespace Registry {

void reg_set_int(const char *name, int val) {
  HKEY hKey;
  unsigned long result;
  int stat;

  stat = RegCreateKeyEx(HKEY_CURRENT_USER, GC_REGISTRY_NAME, 0, "gc_key", 0,
                        KEY_WRITE, NULL, &hKey, &result);

  stat = RegSetValueEx(hKey, name, 0, REG_DWORD, (LPBYTE)&val, sizeof(DWORD));

  stat = RegCloseKey(hKey);
}

int reg_get_int(const char *name, int *val) {
  HKEY hKey;
  unsigned long datatype = REG_DWORD;
  DWORD buffersize = sizeof(DWORD);
  int stat;

  stat = RegOpenKeyEx(HKEY_CURRENT_USER, GC_REGISTRY_NAME, 0, KEY_READ, &hKey);
  if (stat)
    return (stat);

  stat = RegQueryValueEx(hKey, name, 0, &datatype, (LPBYTE)val, &buffersize);

  RegCloseKey(hKey);

  return (stat);
}

void reg_set_string(const char *name, const char *string) {
  HKEY hKey;
  unsigned long result;
  DWORD stat;

  stat = RegCreateKeyEx(HKEY_CURRENT_USER, GC_REGISTRY_NAME, 0, "gc_key", 0,
                        KEY_WRITE, NULL, &hKey, &result);

  stat = RegSetValueEx(hKey, name, 0, REG_SZ, (LPBYTE)string,
                       (DWORD)(strlen(string) + 1));

  stat = RegCloseKey(hKey);
}

int reg_get_string(const char *name, char *string, int maxlen) {
  HKEY hKey;
  unsigned long datatype = REG_SZ;
  DWORD buffersize = maxlen;
  int stat;

  /* Open registry key. */
  stat = RegOpenKeyEx(HKEY_CURRENT_USER, GC_REGISTRY_NAME, 0, KEY_READ, &hKey);
  if (stat)
    return (stat);

  stat = RegQueryValueEx(hKey, name, 0, &datatype, (LPBYTE)string, &buffersize);

  /* Close registry key. */
  RegCloseKey(hKey);

  return (stat);
}
void set_enable_wld(bool enable) {
  const auto temp = static_cast<int>(enable);
  reg_set_int("DarkHorse_enable_wld", temp);
}
void set_max_db_pieces(int val) { reg_set_int("DarkHorse_max_db_pieces", val); }
void set_db_size(int val) { reg_set_int("DarkHorse_db_size", val); }
void set_hash_size(int val) { reg_set_int("DarkHorse_hash_size", val); }
void set_db_path(std::string path) {
  reg_set_string("DarkHorse_db_path", path.c_str());
}

std::optional<int> get_db_size() {
  int value;
  if (reg_get_int("DarkHorse_db_size", &value) == 0) {
    return std::make_optional(value);
  }
  return std::nullopt;
}

bool use_wld_db() {
  int value;
  if (reg_get_int("DarkHorse_enable_wld", &value) == 0) {
    return value != 0;
  }
  return false;
}

std::optional<int> get_hash_size() {
  int value;
  if (reg_get_int("DarkHorse_hash_size", &value) == 0) {
    return std::make_optional(value);
  }
  return std::nullopt;
}

std::optional<int> get_max_db_pieces() {
  int value;
  if (reg_get_int("DarkHorse_max_db_pieces", &value) == 0) {
    return std::make_optional(value);
  }
  return std::nullopt;
}
std::optional<std::string> get_db_path() {
  char data[REG_MSG_SIZE];
  if (reg_get_string("DarkHorse_db_path", data, REG_MSG_SIZE) == 0) {
    return std::make_optional(std::string(data));
  }
  return std::nullopt;
}

} // namespace Registry
