//move sample definition from dataloader here
//should make stuff a little easier to handle :)
use crate::Pos::Position;
use byteorder::WriteBytesExt;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::prelude::*;
#[derive(Debug)]
pub enum SampleType {
    Fen(String),   //a not yet converted FenString
    Pos(Position), //already converted to a position
    None,
}

#[derive(Debug, PartialEq)]
pub enum Result {
    UNKNOWN,
    WIN,
    LOSS,
    DRAW,
}

impl Default for Result {
    fn default() -> Self {
        Result::UNKNOWN
    }
}

impl Default for SampleType {
    fn default() -> Self {
        SampleType::None
    }
}

impl From<i8> for Result {
    fn from(item: i8) -> Self {
        match item {
            1 => Result::LOSS,
            2 => Result::WIN,
            3 => Result::DRAW,
            _ => Result::UNKNOWN,
        }
    }
}

impl From<&str> for Result {
    fn from(item: &str) -> Self {
        match item {
            "loss" | "LOSS" | "LOST" | "lost" => Result::LOSS,
            "win" | "WIN" | "WON" | "won" => Result::WIN,
            "DRAW" | "draw" => Result::DRAW,
            _ => Result::UNKNOWN,
        }
    }
}

#[derive(Default, Debug)]
pub struct Sample {
    pub position: SampleType,
    pub eval: i16,
    pub result: Result,
}

impl Sample {
    pub fn write_fen<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        if let SampleType::Fen(ref fen_string) = self.position {
            let length: u16 = fen_string.len() as u16;
            writer.write_u16::<LittleEndian>(length)?;
            writer.write_all(fen_string.as_bytes())?;
            writer.write_i16::<LittleEndian>(self.eval)?;
            let conv = match self.result {
                Result::LOSS => 1,
                Result::WIN => 2,
                Result::DRAW => 3,
                Result::UNKNOWN => 0,
            };
            writer.write_i8(conv)?;
        }

        Ok(())
    }

    pub fn read_into<R: Read>(&mut self, reader: &mut R) -> std::io::Result<()> {
        // to be added
        let length: u16 = reader.read_u16::<LittleEndian>()?;
        let mut buffer = vec![0; length as usize];
        reader.read_exact(&mut buffer)?;
        self.position = SampleType::Fen(String::from_utf8(buffer).unwrap());
        self.eval = reader.read_i16::<LittleEndian>()?;
        let conv = reader.read_i8()?;
        self.result = match conv {
            1 => Result::LOSS,
            2 => Result::WIN,
            3 => Result::DRAW,
            0 => Result::UNKNOWN,
            _ => Result::UNKNOWN,
        };

        Ok(())
    }
}

impl From<(Position, Result)> for Sample {
    fn from(value: (Position, Result)) -> Self {
        //needs to be implemented
        //see Data.rs for code
        Sample::default()
    }
}
