use std::io::ErrorKind;

use bloomfilter::reexports::bit_vec::BitBlock;

const BLACK: i8 = -1;
const WHITE: i8 = 1;
//below needed for conversion to other bitboard layout
const MASK_L3: u32 = 14737632;
const MASK_L5: u32 = 117901063;
const MASK_R3: u32 = 117901056;
const MASK_R5: u32 = 3772834016;

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

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
//squares should be 0-index (where fen_strings are 1-index)
pub enum Square {
    BPAWN(u8),
    WPAWN(u8),
    BKING(u8),
    WKING(u8),
}

const BRANK_BLACK: u32 = (1 << 18) | (1 << 12) | (1 << 6) | (1 << 0);
const BRANK_WHITE: u32 = (1 << 11) | (1 << 5) | (1 << 31) | (1 << 25);

const NOT_RL_7: u32 = (1 << 18) | (1 << 26) | (1 << 2) | (1 << 10);
const NOT_RR_1: u32 = NOT_RL_7;
const NOT_RL_1: u32 = (1 << 1) | (1 << 9) | (1 << 17) | (1 << 25);
const NOT_RR_7: u32 = NOT_RL_1;

const MASK_COL_1: u32 = 286331153;
const MASK_COL_2: u32 = 572662306;
const MASK_COL_3: u32 = 1145324612;
const MASK_COL_4: u32 = 2290649224;
#[derive(PartialEq, Default, Clone, Copy, Hash, Eq, Debug)]
pub struct Position {
    pub bp: u32,
    pub wp: u32,
    pub k: u32,
    pub color: i8,
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
        let index = trailing as u8;
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

impl Square {
    pub fn invert(&self) -> Square {
        let inv_sq = |sq: &u8| -> u8 {
            let row = sq / 4;
            let col = sq % 4;
            4 * (7 - row) + 3 - col
        };
        match self {
            Square::BPAWN(ind) => Square::WPAWN(inv_sq(ind)),
            Square::WPAWN(ind) => Square::BPAWN(inv_sq(ind)),
            Square::BKING(ind) => Square::WKING(inv_sq(ind)),
            Square::WKING(ind) => Square::BKING(inv_sq(ind)),
        }
    }
}

#[derive(PartialEq, Debug, Default, Copy, Clone)]
pub struct Move {
    from: u32,
    to: u32,
    captures: u32,
}
pub struct MoveList {
    pub moves: [Move; 40],
    pub length: usize,
}

impl Move {
    pub fn get_from_index(&self) -> u32 {
        return self.from.trailing_zeros();
    }

    pub fn get_to_index(&self) -> u32 {
        return self.to.trailing_zeros();
    }

    pub fn get_board_from_index(&self) -> usize {
        return BOARD_BIT[self.get_from_index() as usize];
    }

    pub fn get_board_to_index(&self) -> usize {
        return BOARD_BIT[self.get_to_index() as usize];
    }

    pub fn get_move_encoding(&self) -> usize {
        //for now I can only do that for white to move
        let board_from = 1 << self.get_board_from_index();
        let board_to = 1 << self.get_board_to_index();

        let mut dir: usize = 0;
        if ((((board_from & MASK_L3) << 3) == board_to)
            || (((board_from & MASK_L5) << 5) == board_to))
        {
            dir = 0;
        } else if (((board_from) << 4) == board_to) {
            dir = 1;
        } else if (((board_from) >> 4) == board_to) {
            dir = 2;
        } else if ((((board_from & MASK_R3) >> 3) == board_to)
            || (((board_from & MASK_R5) >> 5) == board_to))
        {
            dir = 3;
        };

        return 4 * self.get_board_from_index() + dir;
    }
}

fn move_left<const COLOR: i8>(maske: u32) -> u32 {
    if COLOR == BLACK {
        (maske & (!NOT_RL_7) & (!BRANK_WHITE)).rotate_left(7)
    } else {
        (maske & (!NOT_RR_7) & (!BRANK_BLACK)).rotate_right(7)
    }
}

fn move_right<const COLOR: i8>(maske: u32) -> u32 {
    if COLOR == BLACK {
        (maske & (!NOT_RL_1) & (!BRANK_WHITE)).rotate_left(1)
    } else {
        (maske & (!NOT_RR_1) & (!BRANK_BLACK)).rotate_right(1)
    }
}

fn get_horizontal_flip(b: u32) -> u32 {
    let mut x: u32 = (b & MASK_COL_4) >> 3u32;
    x |= (b & MASK_COL_3) >> 1u32;
    x |= (b & MASK_COL_1) << 3u32;
    x |= (b & MASK_COL_2) << 1u32;
    return x;
}

fn get_vertical_flip(b: u32) -> u32 {
    let mut x: u32 = b >> 28u32;
    x |= (b >> 20u32) & 0xf0u32;
    x |= (b >> 12u32) & 0xf00u32;
    x |= (b >> 4u32) & 0xf000u32;

    x |= b << 28u32;
    x |= (b << 20u32) & 0x0f000000u32;
    x |= (b << 12u32) & 0x00f00000u32;
    x |= (b << 4u32) & 0x000f0000u32;
    return x;
}

fn get_mirrored(b: u32) -> u32 {
    return get_horizontal_flip(get_vertical_flip(b));
}

impl Position {
    pub fn print_position(&self) {
        for i in (0..8).rev() {
            for j in (0..4).rev() {
                let maske: u32 = 1u32 << (4 * i + j);
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
        if (m.from & self.k) != 0 {
            self.k |= m.to;
            self.k &= !m.from;
        }
        self.k &= !m.captures;

        if self.color == BLACK {
            self.bp &= !m.from;
            self.bp |= m.to;
            self.wp &= !m.captures;
            if (m.to & BRANK_WHITE) != 0 {
                self.k |= m.to;
            }
        } else {
            self.wp &= !m.from;
            self.wp |= m.to;
            self.bp &= !m.captures;
            if (m.to & BRANK_BLACK) != 0 {
                self.k |= m.to;
            }
        }
        self.color = -self.color;

        //need to check for promotion
        //and other things
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

    pub fn get_pieces<const COLOR: i8>(&self) -> u32 {
        if COLOR == -1 {
            self.bp
        } else {
            self.wp
        }
    }

    pub fn piece_count(&self) -> u32 {
        (self.bp.count_ones() + self.wp.count_ones())
    }

    pub fn get_movers<const COLOR: i8>(&self) -> u32 {
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

    pub fn get_jumpers<const COLOR: i8>(&self) -> u32 {
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
            start.bp |= 1 << i;
        }
        for i in 20..32 {
            start.wp |= 1 << i;
        }
        start.k = 0u32;
        return start;
    }

    pub fn iter(&self) -> PosIterator {
        PosIterator {
            partial: self.clone(),
        }
    }

    pub fn get_color_flip(&self) -> Position {
        let mut next = Position::empty();
        next.bp = get_mirrored(self.bp);
        next.wp = get_mirrored(self.wp);
        next.k = get_mirrored(self.k);
        if self.color == -1 {
            next.color = 1;
        } else {
            next.color = -1;
        }
        return next;
    }
}

impl TryFrom<&str> for Position {
    type Error = std::io::Error;

    fn try_from(test: &str) -> std::io::Result<Position> {
        let mut pos: Position = Position::default();

        let add_sq = |pos: &mut Position, color: i32, square: usize| {
            if color == -1 {
                pos.bp |= 1 << (square - 1);
            } else {
                pos.wp |= 1 << (square - 1);
            }
        };

        pos.color = match test.chars().next() {
            Some('W') => 1,
            Some('B') => -1,
            _ => {
                return Err(std::io::Error::new(
                    ErrorKind::NotFound,
                    format!("Error parsing color when fen_string is {}", test),
                ))
            }
        };

        //to be continued
        //need to convert the option in next.unwrap() to a Result

        for s in test.split(":").skip(1) {
            let mut color: i32 = 0;
            let token_op = s.chars().next();
            if token_op == None {
                std::io::Error::new(
                    ErrorKind::NotFound,
                    format!("Error parsing color when fen_string is {}", test),
                );
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
                        pos.k |= 1 << (square - 1);
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

fn jump_left<const COLOR: i8, const OPP: i8>(
    from: u32,
    captures: u32,
    pos: Position,
) -> (u32, u32) {
    let opp = pos.get_pieces::<OPP>() & !captures;
    let nocc = !(pos.bp | pos.wp);
    let captured = move_left::<COLOR>(from) & opp;
    (captured, move_left::<COLOR>(captured) & nocc)
}

fn jump_right<const COLOR: i8, const OPP: i8>(
    from: u32,
    captures: u32,
    pos: Position,
) -> (u32, u32) {
    let opp = pos.get_pieces::<OPP>() & !captures;
    let nocc = !(pos.bp | pos.wp);
    let captured = move_right::<COLOR>(from) & opp;
    (captured, move_right::<COLOR>(captured) & nocc)
}

fn add_capture<const COLOR: i8, const OPP: i8, const is_king: bool>(
    orig: u32,
    from: u32,
    captures: u32,
    pos: Position,
    liste: &mut MoveList,
) {
    //handling pawn captures first
    let mut dest: u32 = 0;
    let left_cap = jump_left::<COLOR, OPP>(from, captures, pos);
    if left_cap.1 != 0 {
        add_capture::<COLOR, OPP, is_king>(orig, left_cap.1, captures | left_cap.0, pos, liste);
    }
    let right_cap = jump_right::<COLOR, OPP>(from, captures, pos);
    if right_cap.1 != 0 {
        add_capture::<COLOR, OPP, is_king>(orig, right_cap.1, captures | right_cap.0, pos, liste);
    }
    dest |= left_cap.1 | right_cap.1;
    if is_king {
        let king_left = jump_left::<OPP, OPP>(from, captures, pos);
        if king_left.1 != 0 {
            add_capture::<COLOR, OPP, is_king>(
                orig,
                king_left.1,
                captures | king_left.0,
                pos,
                liste,
            );
        }

        let king_right = jump_right::<OPP, OPP>(from, captures, pos);
        if king_right.1 != 0 {
            add_capture::<COLOR, OPP, is_king>(
                orig,
                king_right.1,
                captures | king_right.0,
                pos,
                liste,
            );
        }
        dest |= king_left.1 | king_right.1;
    }
    if dest == 0 {
        liste.add_move(orig, from, captures);
    }
}

impl MoveList {
    pub fn new() -> MoveList {
        MoveList {
            length: 0,
            moves: [Move::default(); 40],
        }
    }

    fn add_move(&mut self, from: u32, to: u32, captures: u32) {
        let scrap: usize = (to == 0) as usize;
        self.moves[self.length + scrap] = Move {
            from,
            to,
            captures: captures,
        };
        self.length += (to != 0) as usize;
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

    pub fn get_silent_movers<const COLOR: i8, const OPP: i8>(&mut self, pos: &Position) {
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

    pub fn get_captures<const COLOR: i8, const OPP: i8>(&mut self, pos: &mut Position) {
        let jumpers = pos.get_jumpers::<COLOR>();
        let mut pawns = jumpers & !pos.k;
        let mut kings = jumpers & pos.k;
        while pawns != 0 {
            let from = pawns & !(pawns - 1u32);
            add_capture::<COLOR, OPP, false>(from, from, 0, pos.clone(), self);
            pawns &= pawns - 1;
        }
        while kings != 0 {
            let from = kings & !(kings - 1u32);
            if COLOR == BLACK {
                pos.bp ^= from;
            } else {
                pos.wp ^= from;
            }
            add_capture::<COLOR, OPP, true>(from, from, 0, pos.clone(), self);
            if COLOR == BLACK {
                pos.bp ^= from;
            } else {
                pos.wp ^= from;
            }
            kings &= kings - 1;
        }
    }

    pub fn get_moves(&mut self, pos: Position) {
        let mut copy = pos.clone();
        if pos.color == BLACK {
            if pos.get_jumpers::<-1>() != 0 {
                self.get_captures::<BLACK, WHITE>(&mut copy);
                return;
            }
            self.get_silent_movers::<BLACK, WHITE>(&pos);
        } else {
            if pos.get_jumpers::<1>() != 0 {
                self.get_captures::<WHITE, BLACK>(&mut copy);
                return;
            }
            self.get_silent_movers::<WHITE, BLACK>(&pos);
        }
    }

    pub fn iter(&mut self) -> std::slice::Iter<'_, Move> {
        (&self.moves[0..self.length]).iter()
    }
}
