#include <CheckerBoard.h>
#include <fstream>
std::ofstream debug("debug.file",std::ios::app);
extern"C" int getmove
(
	int board[8][8],
	int color,
	double maxtime,
	char str[1024],
	int* playnow,
	int info,
	int moreinfo,
	struct CBmove* move
){
    //to be implemented

	if(info & CB_RESET_MOVES){
		//debug<<"Reset moves"<<"\n";
		game_board = Board{};
		game_board = Position::get_start_position();
		TT.age_counter=0;
	}
	//setting the board
	
	Position temp;
	for(auto i=0;i<32;++i){
		const auto cb_index = To64[i];
		int row = cb_index/8;
		int col = 7-cb_index%8;
		const auto p_square = board[col][row];
		if((p_square==(CB_BLACK|CB_KING))){
			temp.BP|=1u<<i;
			temp.K|=1u<<i;
		}
		if((p_square==(CB_WHITE|CB_KING))){
			temp.WP|=1u<<i;
			temp.K|=1u<<i;
		}
		if((p_square==(CB_BLACK|CB_MAN))){
			temp.BP|=1u<<i;
		}
		if((p_square==(CB_WHITE|CB_MAN))){
			temp.WP|=1u<<i;
		}
		if((p_square==(CB_FREE))){
			temp.WP&=~(1u<<i);
			temp.BP&=~(1u<<i);
			temp.K&=~(1u<<i);
		}
	}
	temp.color =(color == CB_BLACK)?BLACK : WHITE;
	//CheckerBoard Bug
		auto m = Position::get_move(game_board.get_position(),temp);
		if(m.has_value()){
			//debug<<"Move played by opponent"<<std::endl;
			//debug<<"From: "<<m->get_from_index()<<" To: "<<m->get_to_index()<<std::endl;
			game_board.play_move(m.value());
		}else{
			//debug<<"Reset move, because we didnt find it"<<std::endl;
			
		game_board = Board{};
		game_board = temp;
		TT.age_counter=0;
		}
	
	

		if(!engine_initialized){
			  use_classical(true);
			//debug<<"Initialized engine"<<std::endl;
			TT.resize(21);
		init_tablebase(2000,6,debug);
		initialize();
		Statistics::mPicker.init();
		engine_initialized=true;
	}
	uint32_t time_to_use = static_cast<int>(maxtime*1000);
	//debug<<"Time to use: " << time_to_use<<"\n";
	//debug<<"Fen: "<<game_board.get_position().get_fen_string()<<std::endl;
	Move best;
    auto value = searchValue(game_board, best, MAX_PLY, time_to_use, true,debug);
	//debug<<"\n";
	//debug<<"Found the move: "<<" From: "<<best.get_from_index()<< " To: "<<best.get_to_index()<<std::endl;
	game_board.play_move(best); 

/* 	MoveListe liste;
	get_moves(game_board.get_position(),liste);
	game_board.make_move(liste[0]);
	last_position = game_board.get_position(); */

	//update the board
	
	Position c = game_board.get_position();
	for(auto i=0;i<32;++i){
		const auto mask = 1<<i;
		size_t cb_index = To64[i];
		int row = cb_index/8;
		int col = 7-cb_index%8;

		if((c.BP&c.K&mask)){
			board[col][row]=CB_BLACK|CB_KING;
			
			
		}else if((c.WP&c.K&mask)){
			board[col][row]=CB_WHITE|CB_KING;
			
		}else if((c.BP&mask)){
			board[col][row]=CB_BLACK|CB_MAN;
			
			
		}else if((c.WP&mask)){
			board[col][row]=CB_WHITE|CB_MAN;
			
		}else{
			board[col][row]=CB_FREE;
		}
	}
	//debug<<"Position after we moved"<<"\n";
	//debug<<game_board.get_position().get_pos_string();

	if(isMateVal(value)){
		return (value<0)?CB_LOSS : CB_WIN;
	}
	last_position = game_board.get_position();
	return CB_UNKNOWN;
}

int enginecommand(char str[256], char reply[1024]){
    const int REPLY_MAX = 1024;
	char command[256], param1[256], param2[256];
	char* stopstring;

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

	if (strcmp(command, "staticevaluation") == 0) {
		Move best;
		//auto value = searchValue(game_board,best, 0, 100000000, true,);
		//static_eval(param1, reply);
		return(1);
	}

/* 	if (strcmp(param1, "check_wld_dir") == 0) {
		check_wld_dir(param2, reply);
		return(1);
	} */

	if (strcmp(command, "set") == 0) {
		int val;

		if (strcmp(param1, "hashsize") == 0) {

			int numMBs = strtol(param2, &stopstring, 10);
			//for now just use some default value
			
			TT.resize(21);
			return 1;
		}
 
	 /* 	if (strcmp(param1, "dbpath") == 0) {
			char* p = strstr(str, "dbpath");	
			while (!isspace(*p))			
				++p;
			while (isspace(*p))
				++p;
			if (strcmp(p, checkerBoard.db_path)) {
				checkerBoard.request_egdb_init = true;
				strcpy(checkerBoard.db_path, p);
				save_dbpath(checkerBoard.db_path);
			}

			sprintf(reply, "dbpath set to %s", checkerBoard.db_path);
			return(1);
		}   */
/* 
		if (strcmp(param1, "enable_wld") == 0) {
			val = strtol(param2, &stopstring, 10);
			if (val != checkerBoard.enable_wld) {
				checkerBoard.request_egdb_init = true;
				checkerBoard.enable_wld = val;
				save_enable_wld(checkerBoard.enable_wld);
			}

			snprintf(reply, REPLY_MAX, "enable_wld set to %d", checkerBoard.enable_wld);
			return(1);
		} */

	/* 	if (strcmp(param1, "book") == 0) {
			val = strtol(param2, &stopstring, 10);
			if (val != checkerBoard.useOpeningBook) {
				checkerBoard.useOpeningBook = val;
				save_book_setting(checkerBoard.useOpeningBook);
			}

			snprintf(reply, REPLY_MAX, "book set to %d", checkerBoard.useOpeningBook);
			return(1);
		} */
	/* 	if (strcmp(param1, "max_dbpieces") == 0) {
			val = strtol(param2, &stopstring, 10);
			if (val != checkerBoard.max_dbpieces) {
				checkerBoard.request_egdb_init = true;
				checkerBoard.max_dbpieces = val;
				save_max_dbpieces(checkerBoard.max_dbpieces);
			}

			sprintf(reply, "max_dbpieces set to %d", checkerBoard.max_dbpieces);
			return(1);
		}
 */
	/* 	if (strcmp(param1, "dbmbytes") == 0) {
			val = strtol(param2, &stopstring, 10);
			if (val != checkerBoard.wld_cache_mb) {
				checkerBoard.request_egdb_init = true;
				checkerBoard.wld_cache_mb = val;
				save_dbmbytes(checkerBoard.wld_cache_mb);
			}

			sprintf(reply, "dbmbytes set to %d", checkerBoard.wld_cache_mb);
			return(1);
		}
	} */

	if (strcmp(command, "get") == 0) {
		/* if (strcmp(param1, "hashsize") == 0) {
			get_hashsize(&engine.TTable.sizeMb);
			snprintf(reply, REPLY_MAX, "%d", engine.TTable.sizeMb);
			return 1;
		} */

		if (strcmp(param1, "protocolversion") == 0) {
			snprintf(reply, REPLY_MAX, "2");
			return 1;
		}

		if (strcmp(param1, "gametype") == 0) {
			snprintf(reply, REPLY_MAX, "%d", GT_ENGLISH);
			return 1;
		}

		/* if (strcmp(param1, "dbpath") == 0) {
			get_dbpath(checkerBoard.db_path, sizeof(checkerBoard.db_path));
			snprintf(reply, REPLY_MAX, checkerBoard.db_path);
			return(1);
		}

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
	}

	strcpy(reply, "?");
	return 0;
} 