use std::io::ErrorKind;

const BLACK: i32 = -1;
const WHITE: i32 = 1;
/*
const BIT_BOARD: [u32; 32] = [
    18, 12, 6, 0, 19, 13, 7, 1, 26, 20, 14, 8, 27, 21, 15, 9, 2, 28, 22, 16, 3, 29, 23, 17, 10, 4,
    30, 24, 11, 5, 31, 25,
];
*/
/*
  11  05  31  25
10  04  30  24
  03  29  23  17
02  28  22  16
  27  21  15  09
26  20  14  08
  19  13  07  01
18  12  06  00
*/
//reordering to be continued
pub const BIT_BOARD: [usize; 32] = [
    0, 6, 12, 18, 1, 7, 13, 19, 8, 14, 20, 26, 9, 15, 21, 27, 16, 22, 28, 2, 17, 23, 29, 3, 24, 30,
    4, 10, 25, 31, 5, 11,
];

pub const BOARD_BIT: [usize; 32] = [
    0, 4, 19, 23, 26, 30, 1, 5, 8, 12, 27, 31, 2, 6, 9, 13, 16, 20, 3, 7, 10, 14, 17, 21, 24, 28,
    11, 15, 18, 22, 25, 29,
];

#[derive(Debug, Clone, Copy)]
pub enum Square {
    BPAWN(usize),
    WPAWN(usize),
    BKING(usize),
    WKING(usize),
}

const BRANK_BLACK: u32 = (1 << 18) | (1 << 12) | (1 << 6) | (1 << 0);
const BRANK_WHITE: u32 = (1 << 11) | (1 << 5) | (1 << 31) | (1 << 25);

const NOT_RL_7: u32 = (1 << 18) | (1 << 26) | (1 << 2) | (1 << 10);
const NOT_RR_1: u32 = NOT_RL_7;
const NOT_RL_1: u32 = (1 << 1) | (1 << 9) | (1 << 17) | (1 << 25);
const NOT_RR_7: u32 = NOT_RL_1;

#[derive(PartialEq, Default, Clone, Copy, Hash, Eq, Debug)]
pub struct Position {
    pub bp: u32,
    pub wp: u32,
    pub k: u32,
    pub color: i32,
}

pub struct PosIterator {
    partial: Position,
}
//to be continued
impl Iterator for PosIterator {
    type Item = Square;
    fn next(&mut self) -> Option<Self::Item> {
        let occ = self.partial.bp | self.partial.wp;

        if occ == 0 {
            return None;
        }
        let trailing = occ.trailing_zeros();
        let lsb = 1u32 << trailing;
        let index = trailing as usize;
        //removing the lsb from partial

        let value: i32 = ((lsb & self.partial.bp) != 0) as i32
            + (((lsb & self.partial.wp) != 0) as i32) * 2i32
            + (((lsb & self.partial.k) != 0) as i32) * 3i32;

        self.partial.bp &= !lsb;
        self.partial.wp &= !lsb;
        self.partial.k &= !lsb;
        match value {
            1 => Some(Square::BPAWN(index)),
            2 => Some(Square::WPAWN(index)),
            4 => Some(Square::BKING(index)),
            5 => Some(Square::WKING(index)),
            _ => None,
        }
    }
}

#[derive(PartialEq, Default, Copy, Clone)]
pub struct Move {
    from: u32,
    to: u32,
    captures: u32,
}
pub struct MoveList {
    pub moves: [Move; 40],
    pub length: usize,
}

fn move_left<const COLOR: i32>(maske: u32) -> u32 {
    if COLOR == BLACK {
        (maske & (!NOT_RL_7) & (!BRANK_WHITE)).rotate_left(7)
    } else {
        (maske & (!NOT_RR_7) & (!BRANK_BLACK)).rotate_right(7)
    }
}

fn move_right<const COLOR: i32>(maske: u32) -> u32 {
    if COLOR == BLACK {
        (maske & (!NOT_RL_1) & (!BRANK_WHITE)).rotate_left(1)
    } else {
        (maske & (!NOT_RR_1) & (!BRANK_BLACK)).rotate_right(1)
    }
}

impl Position {
    pub fn print_position(&self) {
        for i in (0..8).rev() {
            for j in (0..4).rev() {
                let maske: u32 = 1u32 << (BIT_BOARD[4 * i + j]);
                let value: i32 = ((maske & self.bp) != 0) as i32
                    + (((maske & self.wp) != 0) as i32) * 2i32
                    + (((maske & self.k) != 0) as i32) * 3i32;
                if i % 2 == 1 {
                    print!("[ ]");
                }
                match value {
                    1i32 => print!("[0]"),
                    2i32 => print!("[X]"),
                    4i32 => print!("[B]"),
                    5i32 => print!("[W]"),
                    _ => print!("[ ]"),
                }
                if i % 2 == 0 {
                    print!("[ ]");
                }
            }
            println!();
        }
    }

    pub fn make_move(&mut self, m: &Move) {
        if self.color == BLACK {
            self.bp &= !m.from;
            self.bp |= m.to;
            self.wp &= !m.captures;
        } else {
            self.wp &= !m.from;
            self.wp |= m.to;
            self.bp &= !m.captures;
        }
        //need to check for promotion
        //and other things
        if self.k != 0 {}
    }

    pub fn undo_move(&mut self, _m: &Move) {
        //to be implemented
    }

    //need to understand the borrow trait
    pub fn empty() -> Position {
        Position {
            color: BLACK,
            ..Position::default()
        }
    }

    pub fn get_pieces<const COLOR: i32>(&self) -> u32 {
        if COLOR == -1 {
            self.bp
        } else {
            self.wp
        }
    }

    pub fn get_movers<const COLOR: i32>(&self) -> u32 {
        let nocc: u32 = !(self.bp | self.wp);
        let mut movers: u32 = 0;
        if self.k != 0 {
            movers |= move_left::<COLOR>(nocc);
            movers |= move_right::<COLOR>(nocc);
        }
        if COLOR == BLACK {
            movers |= move_left::<WHITE>(nocc);
            movers |= move_right::<WHITE>(nocc);
            movers &= self.bp;
        } else {
            movers |= move_left::<BLACK>(nocc);
            movers |= move_right::<BLACK>(nocc);
            movers &= self.wp;
        }

        return movers;
    }

    pub fn get_jumpers<const COLOR: i32>(&self) -> u32 {
        let nocc: u32 = !(self.bp | self.wp);
        let mut movers: u32 = 0;
        let opp: u32 = if COLOR == BLACK { self.wp } else { self.bp };
        let own: u32 = if COLOR == BLACK { self.bp } else { self.wp };
        if COLOR == BLACK {
            movers |= move_left::<WHITE>(move_left::<WHITE>(nocc) & self.wp);
            movers |= move_right::<WHITE>(move_right::<WHITE>(nocc) & self.wp);
            movers &= self.bp;
        } else {
            movers |= move_left::<BLACK>(move_left::<BLACK>(nocc) & self.bp);
            movers |= move_right::<BLACK>(move_right::<BLACK>(nocc) & self.bp);
            movers &= self.wp;
        }
        if self.k != 0 {
            movers |= move_left::<COLOR>(move_left::<COLOR>(nocc) & opp);
            movers |= move_right::<COLOR>(move_right::<COLOR>(nocc) & opp);
            movers &= own & self.k;
        }
        return movers;
    }

    pub fn get_start_position() -> Position {
        let mut start: Position = Position::empty();
        for i in 0..12 {
            start.bp |= 1 << BIT_BOARD[i];
        }
        for i in 20..32 {
            start.wp |= 1 << BIT_BOARD[i];
        }
        start.k = 0u32;
        return start;
    }

    pub fn iter(&self) -> PosIterator {
        PosIterator {
            partial: self.clone(),
        }
    }
}

impl TryFrom<&str> for Position {
    type Error = std::io::Error;

    fn try_from(test: &str) -> std::io::Result<Position> {
        let mut pos: Position = Position::default();

        let add_sq = |pos: &mut Position, color: i32, square: usize| {
            if color == -1 {
                pos.bp |= 1 << BIT_BOARD[square - 1];
            } else {
                pos.wp |= 1 << BIT_BOARD[square - 1];
            }
        };

        pos.color = match test.chars().next() {
            Some('W') => 1,
            Some('B') => -1,
            _ => {
                return Err(std::io::Error::new(
                    ErrorKind::NotFound,
                    "Error parsing color",
                ))
            }
        };

        //to be continued
        //need to convert the option in next.unwrap() to a Result

        for s in test.split(":").skip(1) {
            let mut color: i32 = 0;
            let token_op = s.chars().next();
            if token_op == None {
                std::io::Error::new(ErrorKind::NotFound, "Error parsing color");
            }
            match token_op {
                Some('W') => color = 1,
                Some('B') => color = -1,
                _ => (),
            }
            let splits = s.split(",");

            for (i, val) in splits.enumerate() {
                let mut sq_str = val.chars();
                if i == 0 {
                    sq_str.next();
                }
                let m = sq_str.clone().next().unwrap();
                match m {
                    'K' => {
                        sq_str.next();
                        let square: usize = match sq_str.as_str().parse() {
                            Ok(n) => n,
                            Err(_) => {
                                return Err(std::io::Error::new(
                                    ErrorKind::NotFound,
                                    "Error parsing squares",
                                ))
                            }
                        };

                        add_sq(&mut pos, color, square);
                        pos.k |= 1 << BIT_BOARD[square - 1];
                    }

                    _ => {
                        let square: usize = match sq_str.as_str().parse() {
                            Ok(n) => n,
                            Err(_) => {
                                return Err(std::io::Error::new(
                                    ErrorKind::NotFound,
                                    "Error parsing squares",
                                ))
                            }
                        };
                        add_sq(&mut pos, color, square);
                    }
                }
            }
        }

        Ok(pos)
    }
}

impl MoveList {
    pub fn empty_list() -> MoveList {
        MoveList {
            length: 0,
            moves: [Move::default(); 40],
        }
    }

    fn add_quiet_move(&mut self, from: u32, to: u32) {
        let scrap: usize = (to == 0) as usize;
        self.moves[self.length + scrap] = Move {
            from,
            to,
            captures: 0,
        };
        self.length += (to != 0) as usize;
    }

    pub fn get_silent_movers<const COLOR: i32, const OPP: i32>(&mut self, pos: &Position) {
        let movers = pos.get_movers::<COLOR>();
        let nocc = !(pos.bp | pos.wp);
        let mut pawns = movers & !pos.k;
        let mut kings = movers & pos.k;
        while pawns != 0 {
            let from = pawns & !(pawns - 1u32);
            self.add_quiet_move(from, move_left::<COLOR>(from) & nocc);
            self.add_quiet_move(from, move_right::<COLOR>(from) & nocc);
            pawns &= pawns - 1;
        }
        while kings != 0 {
            let from = kings & !(kings - 1u32);
            self.add_quiet_move(from, move_left::<COLOR>(from) & nocc);
            self.add_quiet_move(from, move_right::<COLOR>(from) & nocc);
            self.add_quiet_move(from, move_left::<OPP>(from) & nocc);
            self.add_quiet_move(from, move_right::<OPP>(from) & nocc);
            kings &= kings - 1;
        }
    }
    /*
        fn jump_left<const COLOR: i32, const OPP: i32>(from: u32, pos: &Position) -> u32 {
            let opp = pos.get_pieces::<COLOR>();
            let nocc = !(pos.bp | pos.wp);
            move_left::<COLOR>(move_left::<COLOR>(from) & opp) & nocc
        }

        fn jump_right<const COLOR: i32, const OPP: i32>(from: u32, pos: &Position) -> u32 {
            let opp = pos.get_pieces::<COLOR>();
            let nocc = !(pos.bp | pos.wp);
            move_right::<COLOR>(move_right::<COLOR>(from) & opp) & nocc
        }
    */
    fn try_capture<const COLOR: i32, const OPP: i32>(_from: u32, _pos: Position) {
        //need to handle jump loops 'jumps that end at the starting positions'
        //by removing the captures from the position

        //to be continued
    }
    pub fn get_captures<const COLOR: i32, const OPP: i32>(&mut self, pos: Position) {
        let jumpers = pos.get_jumpers::<COLOR>();
        let mut pawns = jumpers & !pos.k;
        let _kings = jumpers & pos.k;
        while pawns != 0 {
            let _from = pawns & !(pawns - 1u32);

            pawns &= pawns - 1;
        }
    }

    pub fn iter(&mut self) -> std::slice::Iter<'_, Move> {
        (&self.moves[0..self.length]).iter()
    }
}
