use std::ops::{Shl, Shr};

use bloomfilter::reexports::bit_vec::BitBlock;
use libc::size_t;

const BLACK: i8 = -1;
const WHITE: i8 = 1;
//below needed for conversion to other bitboard layout
const MASK_L3: u32 = 14737632;
const MASK_L5: u32 = 117901063;
const MASK_R3: u32 = 117901056;
const MASK_R5: u32 = 3772834016;
const MASK_COL_1: u32 = 286331153;
const MASK_COL_2: u32 = 572662306;
const MASK_COL_3: u32 = 1145324612;
const MASK_COL_4: u32 = 2290649224;

const PROMO_SQUARES_WHITE: u32 = 0xf;
const PROMO_SQUARES_BLACK: u32 = 0xf0000000;

pub const BIT_TO_BOARD: [usize; 32] = [
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 27, 26,
    25, 24, 31, 30, 29, 28,
];
pub const BOARD_TO_BIT: [usize; 32] = [
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 27, 26,
    25, 24, 31, 30, 29, 28,
];

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
//squares should be 0-index (where fen_strings are 1-inde/stx)
pub enum Square {
    BPAWN(u8),
    WPAWN(u8),
    BKING(u8),
    WKING(u8),
}

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

    pub fn get_move_encoding(&self) -> i32 {
        let mut dir: i32 = 0;
        if (((self.from & MASK_L3) << 3) == self.to) || (((self.from & MASK_L5) << 5) == self.to) {
            dir = 0;
        } else if ((self.from) << 4) == self.to {
            dir = 1;
        } else if ((self.from) >> 4) == self.to {
            dir = 2;
        } else if ((self.from & MASK_R3) >> 3) == self.to || ((self.from & MASK_R5) >> 5) == self.to
        {
            dir = 3;
        };

        return 4 * self.get_from_index() as i32 + dir;
    }

    pub fn get_move_encoding_from_pos(orig: Position, next: Position) -> Option<i32> {
        //first we need to figure out , what move was played
        let mut liste = MoveList::new();
        liste.get_moves(orig);

        for m in liste.iter() {
            let mut copy = orig;
            copy.make_move(m);
            if copy == next {
                return Some(m.get_move_encoding());
            }
        }
        //At this point there was no move in 'orig' that
        //lead to the position 'next'
        return None;
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
                let board_index = 4 * i + j;
                let bit_index = board_index;
                let maske: u32 = 1u32 << bit_index;
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
            if m.captures != 0 {
                self.wp &= !m.captures;
                self.k &= !m.captures;
            }
            self.bp &= !m.from;
            self.bp |= m.to;

            if ((m.to & PROMO_SQUARES_BLACK) != 0) && ((m.from & self.k) == 0) {
                self.k |= m.to;
            }
        } else {
            if m.captures != 0 {
                self.bp &= !m.captures;
                self.k &= !m.captures;
            }
            self.wp &= !m.from;
            self.wp |= m.to;

            if ((m.to & PROMO_SQUARES_WHITE) != 0) && ((m.from & self.k) == 0) {
                self.k |= m.to;
            }
        }
        if (m.from & self.k) != 0 {
            self.k &= !m.from;
            self.k |= m.to;
        }
        self.color = -self.color;
    }

    pub fn undo_move(&mut self, _m: &Move) {
        //to be implemented
    }

    pub fn get_fen_string(&self) -> String {
        let mut black_pieces_string = String::new();
        let mut white_pieces_string = String::new();

        let mut fen_string = String::new();
        fen_string.push_str(match self.color {
            1 => "W:",
            -1 => "B:",
            _ => "",
        });

        for square in self.iter() {
            match square {
                Square::BPAWN(ind) => {
                    black_pieces_string.push_str(((ind + 1).to_string() + ",").as_str())
                }
                Square::WPAWN(ind) => {
                    white_pieces_string.push_str(((ind + 1).to_string() + ",").as_str())
                }
                Square::WKING(ind) => {
                    white_pieces_string
                        .push_str(("K".to_owned() + (ind + 1).to_string().as_str() + ",").as_str());
                }
                Square::BKING(ind) => {
                    black_pieces_string
                        .push_str(("K".to_owned() + (ind + 1).to_string().as_str() + ",").as_str());
                }
            }
        }

        black_pieces_string = black_pieces_string.trim_end_matches(",").to_string();
        white_pieces_string = white_pieces_string.trim_end_matches(",").to_string();
        fen_string.push_str("W");
        fen_string.push_str(white_pieces_string.as_str());
        fen_string.push_str(":B");
        fen_string.push_str(black_pieces_string.as_str());
        return fen_string.trim_end_matches(",").to_string();
    }

    pub fn empty() -> Position {
        Position {
            color: BLACK,
            ..Position::default()
        }
    }

    pub fn get_own_pieces(self: Position) -> u32 {
        if self.color == BLACK {
            return self.bp;
        } else {
            return self.wp;
        }
    }

    pub fn get_opp_pieces(self: Position) -> u32 {
        if self.color == BLACK {
            return self.wp;
        } else {
            return self.bp;
        }
    }
    pub fn piece_count(&self) -> u32 {
        self.bp.count_ones() + self.wp.count_ones()
    }

    pub fn get_movers(&self) -> u32 {
        let nocc = !(self.wp | self.bp);
        let current = if self.color == BLACK {
            self.bp
        } else {
            self.wp
        };
        let kings = current & self.k;

        let mut movers =
            (default_shift(-self.color, nocc) | forward_mask(-self.color, nocc)) & current;
        if kings != 0 {
            movers |= (default_shift(self.color, nocc) | forward_mask(self.color, nocc)) & kings;
        }
        return movers;
    }

    pub fn get_jumpers<const COLOR: i8>(&self) -> u32 {
        let nocc = !(self.bp | self.wp);
        let current = if COLOR == BLACK { self.bp } else { self.wp };
        let opp = if COLOR == BLACK { self.wp } else { self.bp };
        let kings = current & self.k;

        let mut movers = 0;
        let temp = default_shift(-self.color, nocc) & opp;
        if temp != 0 {
            movers |= forward_mask(-self.color, temp) & current;
        }
        let mut temp = forward_mask(-self.color, nocc) & opp;
        if temp != 0 {
            movers |= default_shift(-self.color, temp) & current;
        }
        if kings != 0 {
            temp = default_shift(self.color, nocc) & opp;
            if temp != 0 {
                movers |= forward_mask(self.color, temp) & kings;
            }
            temp = forward_mask(self.color, nocc) & opp;

            if temp != 0 {
                movers |= default_shift(self.color, temp) & kings;
            }
        }
        return movers;
    }

    pub fn has_capture(&self) -> bool {
        return (self.color == -1 && self.get_jumpers::<-1>() != 0)
            || (self.color == 1 && self.get_jumpers::<1>() != 0);
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
        next.bp = get_mirrored(self.wp);
        next.wp = get_mirrored(self.bp);
        next.k = get_mirrored(self.k);
        next.color = -self.color;
        return next;
    }

    pub fn has_jumps(self) -> bool {
        return (self.get_jumpers::<BLACK>() != 0 && self.color == BLACK)
            || (self.get_jumpers::<WHITE>() != 0 && self.color == WHITE);
    }
}

impl TryFrom<&str> for Position {
    type Error = std::io::Error;

    fn try_from(test: &str) -> std::io::Result<Position> {
        let mut pos: Position = Position::default();

        let add_sq = |pos: &mut Position, color: i32, square: usize| {
            if color == -1 {
                pos.bp |= 1 << square - 1;
            } else {
                pos.wp |= 1 << square - 1;
            }
        };

        pos.color = match test.chars().next() {
            Some('W') => 1,
            Some('B') => -1,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
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
                    std::io::ErrorKind::NotFound,
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
                                    std::io::ErrorKind::NotFound,
                                    "Error parsing squares",
                                ))
                            }
                        };

                        add_sq(&mut pos, color, square);
                        pos.k |= 1 << square - 1;
                    }

                    _ => {
                        let square: usize = match sq_str.as_str().parse() {
                            Ok(n) => n,
                            Err(_) => {
                                return Err(std::io::Error::new(
                                    std::io::ErrorKind::NotFound,
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
    pub fn new() -> MoveList {
        MoveList {
            length: 0,
            moves: [Move::default(); 40],
        }
    }
    pub fn iter(&mut self) -> std::slice::Iter<'_, Move> {
        (&self.moves[0..self.length]).iter()
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

    pub fn get_silent_moves(&mut self, pos: Position) {
        let mut pawn_movers = pos.get_movers() & (!pos.k);
        let mut king_movers = pos.get_movers() & pos.k;

        let nocc = !(pos.bp | pos.wp);

        while king_movers != 0 {
            let maske = king_movers & !(king_movers - 1);
            let mut squares = get_neighbour_squares(pos.color, true, maske);
            squares &= nocc;
            while squares != 0 {
                let next = squares & !(squares - 1);
                self.add_quiet_move(maske, next);
                squares &= squares - 1;
            }
            king_movers &= !maske;
        }

        while pawn_movers != 0 {
            let maske = pawn_movers & !(pawn_movers - 1);
            let mut squares = get_neighbour_squares(pos.color, false, maske);
            squares &= nocc;
            while squares != 0 {
                let next = squares & !(squares - 1);
                self.add_quiet_move(maske, next);
                squares &= squares - 1;
            }
            pawn_movers &= !maske;
        }
    }

    pub fn add_capture(
        &mut self,
        is_king_cap: bool,
        pos: Position,
        orig: u32,
        current: u32,
        captures: u32,
    ) {
        let opp = pos.get_opp_pieces() ^ captures;
        let nocc = !(opp | pos.get_own_pieces());
        let temp0 = default_shift(pos.color, current) & opp;
        let temp1 = forward_mask(pos.color, current) & opp;
        let dest0 = forward_mask(pos.color, temp0) & nocc;
        let dest1 = default_shift(pos.color, temp1) & nocc;

        let mut imed = forward_mask(-pos.color, dest0) | default_shift(-pos.color, dest1);
        let mut dest = dest0 | dest1;
        if is_king_cap {
            let temp2 = default_shift(-pos.color, current) & opp;
            let temp3 = forward_mask(-pos.color, current) & opp;
            let dest2 = forward_mask(-pos.color, temp2) & nocc;
            let dest3 = default_shift(-pos.color, temp3) & nocc;
            imed |= forward_mask(pos.color, dest2) | default_shift(pos.color, dest3);
            dest |= dest2 | dest3;
        }
        if dest == 0 {
            self.add_move(orig, current, captures);
            return;
        }
        while dest != 0 {
            let destMask = dest & !(dest - 1);
            let capMask = imed & !(imed - 1);
            dest &= dest - 1;
            imed &= imed - 1;
            self.add_capture(is_king_cap, pos, orig, destMask, (captures | capMask));
        }
    }
    pub fn loop_captures(&mut self, mut pos: Position) {
        let movers = if pos.color == BLACK {
            pos.get_jumpers::<BLACK>()
        } else {
            pos.get_jumpers::<WHITE>()
        };
        let mut king_jumpers = movers & pos.k;
        let mut pawn_jumpers = movers & (!pos.k);
        while king_jumpers != 0 {
            let maske = king_jumpers & !(king_jumpers - 1);
            if pos.color == BLACK {
                pos.bp ^= maske;
            } else {
                pos.wp ^= maske;
            }
            self.add_capture(true, pos, maske, maske, 0);
            if pos.color == BLACK {
                pos.bp ^= maske;
            } else {
                pos.wp ^= maske;
            }

            king_jumpers &= king_jumpers - 1;
        }

        while pawn_jumpers != 0 {
            let maske = pawn_jumpers & !(pawn_jumpers - 1);
            self.add_capture(false, pos, maske, maske, 0);
            pawn_jumpers &= pawn_jumpers - 1;
        }
    }
    pub fn get_moves(&mut self, pos: Position) {
        if pos.has_jumps() {
            self.loop_captures(pos);
            return;
        }
        self.get_silent_moves(pos);
    }
}
pub fn perft_count(depth: i32, position: Position) -> size_t {
    let mut liste = MoveList::new();
    liste.get_moves(position);

    if depth == 1 {
        return liste.length;
    }

    let mut node_count: size_t = 0;
    for m in liste.iter() {
        let mut cp = position.clone();
        cp.make_move(m);

        node_count += perft_count(depth - 1, cp)
    }

    return node_count;
}
pub fn default_shift(color: i8, maske: u32) -> u32 {
    if color == BLACK {
        return maske << 4;
    } else {
        return maske >> 4;
    }
}

pub fn forward_mask(color: i8, maske: u32) -> u32 {
    if color == BLACK {
        return ((maske & MASK_L3) << 3) | ((maske & MASK_L5) << 5);
    } else {
        return ((maske & MASK_R3) >> 3) | ((maske & MASK_R5) >> 5);
    }
}
pub fn get_neighbour_squares(color: i8, is_king: bool, maske: u32) -> u32 {
    if is_king {
        let mut squares = default_shift(color, maske) | forward_mask(color, maske);
        squares |= forward_mask(-color, maske) | default_shift(-color, maske);
        return squares;
    } else {
        return default_shift(color, maske) | forward_mask(color, maske);
    }
}
