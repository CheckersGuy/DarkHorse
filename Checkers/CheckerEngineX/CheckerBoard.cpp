#include "CheckerBoard.h"
#include "GameLogic.h"
#include "types.h"
bool engine_initialized = false;
Board game_board;

int num_draw_scores = 0;
int hash_size_in_mb = 8;
bool enable_wld = false;
Position previous;
std::string db_path;
#define DB_PATH "E:\\kr_english_wld"
INCBIN(mlh_net, "mlh3.quant");
INCBIN(network, "finalformshuffled.quant");
// INCBIN(network, "oldloss.quant");
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
      tablebase.load_table_base(db_path);
    }
    mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
    network.load_from_array(gnetworkData, gnetworkSize);
    policy.load_from_array(gpolicyData, gpolicySize);
    TT.resize_in_mb(hash_size_in_mb);
    engine_initialized = true;
    glob.reply = str;
    num_draw_scores = 0;

    write_to_logfile("init engine with a hashsize of " +
                     std::to_string(TT.get_size_in_mb()) +
                     " and the db_path: " + db_path);
  }

  // CheckerBoard Bug
  auto m = Position::get_move(game_board.get_position(), temp);

  if (!m.has_value() ||
      (temp.piece_count() > game_board.get_position().piece_count())) {
    // debug << "New Game or Bug" << std::endl;
    //  ISSUE PROBABLY HERE PAY ATTENTION
    TT.clear();
    game_board = Board(temp);
    TT.age_counter = 0;
    num_draw_scores = 0;
  } else if (m.has_value()) {
    TT.age_counter = (TT.age_counter + 1) & 63ull;
    game_board.play_move(m.value());
  }

  uint32_t time_to_use = static_cast<int>(std::round(maxtime * 1000.0));
  Move best;
  auto value =
      searchValue(game_board, best, MAX_PLY, time_to_use, false, std::cout);

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

  if (isWinningEval(std::abs(value))) {
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

  if (strcmp(command, "name") == 0) {
    snprintf(reply, REPLY_MAX, "DarkHorse");
    return 1;
  }

  if (strcmp(command, "about") == 0) {
    snprintf(reply, REPLY_MAX, "Written by Robin Messemer");
    return 1;
  }

  if (strcmp(command, "set") == 0) {
    write_to_logfile("Trying to set anything at all");
    if (strcmp(param1, "hashsize") == 0) {

      const int numMBs = strtol(param2, &stopstring, 10);
      TT.resize_in_mb(numMBs);
      hash_size_in_mb = numMBs;
      write_to_logfile("Trying to set the hashsize");
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
        write_to_logfile("DBPath was set to : " + db_path);
      }

      sprintf(reply, "dbpath set to %s", db_path.c_str());
      return 1;
    }
    // TODO checking if we want to use tablebases
    if (strcmp(param1, "enable_wld") == 0) {
      auto val = strtol(param2, &stopstring, 10);
      write_to_logfile("EnableWldDebug: " + std::to_string(val));
      /*if (val != checkerBoard.enable_wld) {
        checkerBoard.request_egdb_init = true;
        checkerBoard.enable_wld = val;
        save_enable_wld(checkerBoard.enable_wld);
      }
      */

      snprintf(reply, REPLY_MAX, "enable_wld set to %d", val);
      return (1);
    }

    /* 	if (strcmp(param1, "book") == 0) {
                    val = strtol(param2, &stopstring, 10);
                    if (val != checkerBoard.useOpeningBook) {
                            checkerBoard.useOpeningBook = val;
                            save_book_setting(checkerBoard.useOpeningBook);
                    }

                    snprintf(reply, REPLY_MAX, "book set to %d",
       checkerBoard.useOpeningBook); return(1);
            } */
    /* 	if (strcmp(param1, "max_dbpieces") == 0) {
                    val = strtol(param2, &stopstring, 10);
                    if (val != checkerBoard.max_dbpieces) {
                            checkerBoard.request_egdb_init = true;
                            checkerBoard.max_dbpieces = val;
                            save_max_dbpieces(checkerBoard.max_dbpieces);
                    }

                    sprintf(reply, "max_dbpieces set to %d",
       checkerBoard.max_dbpieces); return(1);
            }
*/
    /* 	if (strcmp(param1, "dbmbytes") == 0) {
                    val = strtol(param2, &stopstring, 10);
                    if (val != checkerBoard.wld_cache_mb) {
                            checkerBoard.request_egdb_init = true;
                            checkerBoard.wld_cache_mb = val;
                            save_dbmbytes(checkerBoard.wld_cache_mb);
                    }

                    sprintf(reply, "dbmbytes set to %d",
    checkerBoard.wld_cache_mb); return(1);
            }
    } */
  }
  // GETTING ENGINE INFORMATION
  if (strcmp(command, "get") == 0) {
    write_to_logfile("Get-Command is: " + std::string(param1));
    if (strcmp(param1, "hashsize") == 0) {
      write_to_logfile("Read the hashsize with a value of: " +
                       std::to_string(TT.get_size_in_mb()));
      snprintf(reply, REPLY_MAX, "%d", hash_size_in_mb);
      engine_initialized = false;
      return 1;
    }

    if (strcmp(param1, "protocolversion") == 0) {
      snprintf(reply, REPLY_MAX, "2");
      return 1;
    }

    if (strcmp(param1, "gametype") == 0) {
      snprintf(reply, REPLY_MAX, "%d", GT_ENGLISH);
      write_to_logfile("Giving the gametype: " + std::string(reply));
      return 1;
    }

    if (strcmp(param1, "dbpath") == 0) {
      snprintf(reply, REPLY_MAX, db_path.c_str());
      return 1;
    }
    /*
      if (strcmp(param1, "enable_wld") == 0) {
              get_enable_wld(&checkerBoard.enable_wld);
              snprintf(reply, REPLY_MAX, "%d", checkerBoard.enable_wld);
              return(1);
      }

      if (strcmp(param1, "book") == 0) {
              get_book_setting(&checkerBoard.useOpeningBook);
              snprintf(reply, REPLY_MAX, "%d", checkerBoard.useOpeningBook);
              return(1);
      }

      if (strcmp(param1, "max_dbpieces") == 0) {
              get_max_dbpieces(&checkerBoard.max_dbpieces);
              sprintf(reply, "%d", checkerBoard.max_dbpieces);
              return(1);
      }

      if (strcmp(param1, "dbmbytes") == 0) {
              get_dbmbytes(&checkerBoard.wld_cache_mb);
              sprintf(reply, "%d",checkerBoard.wld_cache_mb);
              return(1);
      } */
  }

  strcpy(reply, "?");
  return 0;
}
