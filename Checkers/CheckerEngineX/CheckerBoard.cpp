#include "CheckerBoard.h"
#include "GameLogic.h"
#include "registry.h"
#include "types.h"
bool engine_initialized = false;
Board game_board;

int num_draw_scores = 0;
int hash_size_in_mb = 8;
int db_size_in_mb = 256;
int max_db_pieces = 6;
bool enable_wld = false;
Position previous;
std::string db_path;
#define DB_PATH "E:\\kr_english_wld"
INCBIN(mlh_net, "mlh3.quant");
INCBIN(network, "registry_128.quant");
INCBIN(policy, "policybigger2.quant");

extern "C" int getmove(int board[8][8], int color, double maxtime,
                       char str[1024], int *playnow, int info, int moreinfo,
                       struct CBmove *move) {
  // to be implemented

  if ((info & CB_RESET_MOVES)) {
    game_board = Board(Position::get_start_position());
    TT.age_counter = 0;
    TT.clear();
    num_draw_scores = 0;
  }

  glob.playnow = playnow;
  glob.reply = str;
  // dunno if this is going to work
  Position temp;
  for (auto i = 0; i < 32; ++i) {
    const auto cb_index = To64[i];
    int row = cb_index / 8;
    int col = 7 - cb_index % 8;
    const auto p_square = board[col][row];
    if ((p_square == (CB_BLACK | CB_KING))) {
      temp.BP |= 1u << i;
      temp.K |= 1u << i;
    }
    if ((p_square == (CB_WHITE | CB_KING))) {
      temp.WP |= 1u << i;
      temp.K |= 1u << i;
    }
    if ((p_square == (CB_BLACK | CB_MAN))) {
      temp.BP |= 1u << i;
    }
    if ((p_square == (CB_WHITE | CB_MAN))) {
      temp.WP |= 1u << i;
    }
    if ((p_square == (CB_FREE))) {
      temp.WP &= ~(1u << i);
      temp.BP &= ~(1u << i);
      temp.K &= ~(1u << i);
    }
  }
  temp.color = (color == CB_BLACK) ? BLACK : WHITE;
  if (!engine_initialized) {
    if (enable_wld) {
      tablebase.num_pieces = max_db_pieces;
      tablebase.cache_size = static_cast<size_t>(db_size_in_mb);
      tablebase.load_table_base(db_path);
      tablebase.reply = str;
    }
    mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
    network.load_from_array(gnetworkData, gnetworkSize);
    policy.load_from_array(gpolicyData, gpolicySize);
    TT.resize_in_mb(hash_size_in_mb);
    engine_initialized = true;
    num_draw_scores = 0;
  }
  // CheckerBoard Bug
  auto m = Position::get_move(game_board.get_position(), temp);

  if (temp.piece_count() > game_board.get_position().piece_count()) {
    TT.clear();
    game_board = Board(temp);
    TT.age_counter = 0;
    num_draw_scores = 0;
    last_eval = -INFINITE; // external variable here
  } else if (m.has_value()) {
    TT.age_counter = TT.age_counter + 1;
    game_board.play_move(m.value());
  } else {
    game_board = Board(temp);
    TT.age_counter = 0;
    num_draw_scores = 0;
    last_eval = -INFINITE;
  }

  uint32_t time_to_use = static_cast<int>(std::round(maxtime * 1000.0 * 0.985));

  Move best;
  // measuring the time it took to find the move
  auto t1 = std::chrono::high_resolution_clock::now();
  auto value =
      searchValue(game_board, best, MAX_PLY, time_to_use, false, std::cout);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = (t2 - t1);
  auto time_searched =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  game_board.play_move(best);
  Position c = game_board.get_position();
  for (auto i = 0; i < 32; ++i) {
    const uint32_t mask = 1u << i;
    size_t cb_index = To64[i];
    int row = cb_index / 8;
    int col = 7 - cb_index % 8;

    if ((c.BP & c.K & mask)) {
      board[col][row] = CB_BLACK | CB_KING;

    } else if ((c.WP & c.K & mask)) {
      board[col][row] = CB_WHITE | CB_KING;

    } else if ((c.BP & mask)) {
      board[col][row] = CB_BLACK | CB_MAN;

    } else if ((c.WP & mask)) {
      board[col][row] = CB_WHITE | CB_MAN;

    } else {
      board[col][row] = CB_FREE;
    }
  }
  if (std::abs(value) >= 700 && isEval(value)) {
    return (value < 0) ? CB_LOSS : CB_WIN;
  }
  return CB_UNKNOWN;
}

int enginecommand(char str[256], char reply[1024]) {
  const int REPLY_MAX = 1024;
  char command[256], param1[256], param2[256];
  char *stopstring;

  command[0] = 0;
  param1[0] = 0;
  param2[0] = 0;
  sscanf(str, "%s %s %s", command, param1, param2);
#ifdef AVX256
  if (strcmp(command, "name") == 0) {
    snprintf(reply, REPLY_MAX, "DarkHorse-avx2 v1.0");
    return 1;
  }
#else
  if (strcmp(command, "name") == 0) {
    snprintf(reply, REPLY_MAX, "DarkHorse v1.0");
    return 1;
  }
#endif

  if (strcmp(command, "about") == 0) {
    snprintf(reply, REPLY_MAX, "Created by Robin Messemer");
    return 1;
  }

  if (strcmp(command, "set") == 0) {
    if (strcmp(param1, "hashsize") == 0) {
      const int numMBs = strtol(param2, &stopstring, 10);
      TT.resize_in_mb(numMBs);
      hash_size_in_mb = numMBs;
      // saving the hash_size in the registry
      Registry::set_hash_size(hash_size_in_mb);
      engine_initialized = false;
      return 1;
    }

    if (strcmp(param1, "dbpath") == 0) {
      char *p = strstr(str, "dbpath");
      while (!isspace(*p))
        ++p;
      while (isspace(*p))
        ++p;
      if (strcmp(p, db_path.c_str())) {
        engine_initialized = false;
        db_path = p;
        // saving the db_path
      }

      Registry::set_db_path(db_path);
      sprintf(reply, "dbpath set to %s", db_path.c_str());
      return 1;
    }
    if (strcmp(param1, "enable_wld") == 0) {
      auto val = strtol(param2, &stopstring, 10);
      if ((val != 0) != enable_wld) {
        engine_initialized = false;
        enable_wld = (val != 0);
      }

      Registry::set_enable_wld(enable_wld);

      snprintf(reply, REPLY_MAX, "enable_wld set to %d", val);
      return 1;
    }
    if (strcmp(param1, "max_dbpieces") == 0) {
      auto val = strtol(param2, &stopstring, 10);
      if (val != max_db_pieces) {
        engine_initialized = false;
        max_db_pieces = val;
      }
      // saving the number of db-pieces in the registry

      Registry::set_max_db_pieces(max_db_pieces);
      sprintf(reply, "max_dbpieces set to %d", max_db_pieces);

      return 1;
    }

    if (strcmp(param1, "dbmbytes") == 0) {
      auto val = strtol(param2, &stopstring, 10);
      if (val != db_size_in_mb) {
        engine_initialized = false;
        db_size_in_mb = val;
      }
      // setting the db_size

      Registry::set_db_size(db_size_in_mb);

      sprintf(reply, "dbmbytes set to %d", db_size_in_mb);
      return 1;
    }
  }
  // GETTING ENGINE INFORMATION
  if (strcmp(command, "get") == 0) {

    if (strcmp(param1, "hashsize") == 0) {
      hash_size_in_mb = Registry::get_hash_size().value_or(hash_size_in_mb);
      snprintf(reply, REPLY_MAX, "%d", hash_size_in_mb);
      return 1;
    }

    if (strcmp(param1, "protocolversion") == 0) {
      snprintf(reply, REPLY_MAX, "2");
      return 1;
    }

    if (strcmp(param1, "gametype") == 0) {
      snprintf(reply, REPLY_MAX, "%d", GT_ENGLISH);
      return 1;
    }

    if (strcmp(param1, "dbpath") == 0) {
      db_path = Registry::get_db_path().value_or(db_path);
      snprintf(reply, REPLY_MAX, db_path.c_str());
      return 1;
    }

    if (strcmp(param1, "enable_wld") == 0) {
      enable_wld = Registry::use_wld_db();
      snprintf(reply, REPLY_MAX, "%d", enable_wld);
      return 1;
    }

    if (strcmp(param1, "max_dbpieces") == 0) {
      max_db_pieces = Registry::get_max_db_pieces().value_or(max_db_pieces);
      sprintf(reply, "%d", max_db_pieces);
      return 1;
    }

    if (strcmp(param1, "dbmbytes") == 0) {
      db_size_in_mb = Registry::get_db_size().value_or(db_size_in_mb);
      sprintf(reply, "%d", db_size_in_mb);
      return 1;
    }
  }
  strcpy(reply, "?");
  return 0;
}
