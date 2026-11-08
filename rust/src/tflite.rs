//! Lecture du strict nécessaire d'un fichier .tflite (FlatBuffers, schéma TensorFlow Lite).

#[derive(Clone, Copy)]
struct Table<'a> {
    data: &'a [u8],
    pos: usize,
}

fn u16_at(data: &[u8], pos: usize) -> usize {
    u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize
}

fn u32_at(data: &[u8], pos: usize) -> usize {
    u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize
}

impl<'a> Table<'a> {
    fn root(data: &'a [u8]) -> Self {
        Table { data, pos: u32_at(data, 0) }
    }

    fn field(&self, id: usize) -> Option<usize> {
        let soffset = i32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        let vtable = (self.pos as i64 - soffset as i64) as usize;
        let slot = 4 + 2 * id;
        if slot >= u16_at(self.data, vtable) {
            return None;
        }
        match u16_at(self.data, vtable + slot) {
            0 => None,
            offset => Some(self.pos + offset),
        }
    }

    fn table(&self, id: usize) -> Option<Table<'a>> {
        self.field(id).map(|p| Table { data: self.data, pos: p + u32_at(self.data, p) })
    }

    /// Retourne (position du 1er élément, nombre d'éléments).
    fn vector(&self, id: usize) -> Option<(usize, usize)> {
        self.field(id).map(|p| {
            let start = p + u32_at(self.data, p);
            (start + 4, u32_at(self.data, start))
        })
    }

    fn tables(&self, id: usize) -> Vec<Table<'a>> {
        let (start, len) = self.vector(id).unwrap_or((0, 0));
        (0..len)
            .map(|i| {
                let p = start + 4 * i;
                Table { data: self.data, pos: p + u32_at(self.data, p) }
            })
            .collect()
    }

    fn i32s(&self, id: usize) -> Vec<i32> {
        let (start, len) = self.vector(id).unwrap_or((0, 0));
        (0..len).map(|i| u32_at(self.data, start + 4 * i) as u32 as i32).collect()
    }

    fn f32s(&self, id: usize) -> Vec<f32> {
        let (start, len) = self.vector(id).unwrap_or((0, 0));
        (0..len).map(|i| f32::from_bits(u32_at(self.data, start + 4 * i) as u32)).collect()
    }

    fn i64s(&self, id: usize) -> Vec<i64> {
        let (start, len) = self.vector(id).unwrap_or((0, 0));
        (0..len)
            .map(|i| i64::from_le_bytes(self.data[start + 8 * i..start + 8 * i + 8].try_into().unwrap()))
            .collect()
    }

    fn bytes(&self, id: usize) -> &'a [u8] {
        match self.vector(id) {
            Some((start, len)) => &self.data[start..start + len],
            None => &[],
        }
    }

    fn scalar_i32(&self, id: usize, default: i32) -> i32 {
        self.field(id).map_or(default, |p| u32_at(self.data, p) as u32 as i32)
    }

    fn scalar_u32(&self, id: usize, default: u32) -> u32 {
        self.field(id).map_or(default, |p| u32_at(self.data, p) as u32)
    }

    fn scalar_i8(&self, id: usize, default: i8) -> i8 {
        self.field(id).map_or(default, |p| self.data[p] as i8)
    }
}

pub struct Tensor<'a> {
    pub shape: Vec<usize>,
    pub data: &'a [u8],
    pub scale: Vec<f32>,
    pub zero_point: Vec<i64>,
}

pub struct Operator {
    pub code: i32,
    pub inputs: Vec<i32>,
    pub outputs: Vec<i32>,
    /// Padding, pas, activation... Les numéros de champ dépendent de l'opérateur.
    options: Option<OptionsPos>,
}

#[derive(Clone, Copy)]
struct OptionsPos(usize);

pub struct Model<'a> {
    data: &'a [u8],
    pub tensors: Vec<Tensor<'a>>,
    pub operators: Vec<Operator>,
}

pub const CONV_2D: i32 = 3;
pub const DEPTHWISE_CONV_2D: i32 = 4;
pub const FULLY_CONNECTED: i32 = 9;
pub const LOGISTIC: i32 = 14;
pub const MEAN: i32 = 40;

impl<'a> Model<'a> {
    pub fn parse(data: &'a [u8]) -> Self {
        let root = Table::root(data);
        let buffers = root.tables(4);
        let codes: Vec<i32> = root
            .tables(1)
            .iter()
            .map(|c| (c.scalar_i8(0, 0) as i32).max(c.scalar_i32(3, 0)))
            .collect();
        let graph = root.tables(2)[0];
        let tensors = graph
            .tables(0)
            .iter()
            .map(|t| {
                let quant = t.table(4);
                Tensor {
                    shape: t.i32s(0).iter().map(|&d| d as usize).collect(),
                    data: buffers[t.scalar_u32(2, 0) as usize].bytes(0),
                    scale: quant.map_or(vec![], |q| q.f32s(2)),
                    zero_point: quant.map_or(vec![], |q| q.i64s(3)),
                }
            })
            .collect();
        let operators = graph
            .tables(3)
            .iter()
            .map(|o| Operator {
                code: codes[o.scalar_u32(0, 0) as usize],
                inputs: o.i32s(1),
                outputs: o.i32s(2),
                options: o.table(4).map(|t| OptionsPos(t.pos)),
            })
            .collect();
        Model { data, tensors, operators }
    }

    fn options(&self, op: &Operator) -> Table<'a> {
        Table { data: self.data, pos: op.options.expect("options d'opérateur absentes").0 }
    }

    /// Conv2DOptions : (padding SAME ?, pas h, pas l, activation).
    pub fn conv_options(&self, op: &Operator) -> (bool, usize, usize, i8) {
        let o = self.options(op);
        assert!(o.scalar_i32(4, 1) == 1 && o.scalar_i32(5, 1) == 1, "dilatation non gérée");
        (o.scalar_i8(0, 0) == 0, o.scalar_i32(2, 0) as usize, o.scalar_i32(1, 0) as usize, o.scalar_i8(3, 0))
    }

    /// DepthwiseConv2DOptions : (padding SAME ?, pas h, pas l, activation).
    pub fn depthwise_options(&self, op: &Operator) -> (bool, usize, usize, i8) {
        let o = self.options(op);
        assert!(o.scalar_i32(3, 0) == 1, "multiplicateur de profondeur non géré");
        assert!(o.scalar_i32(5, 1) == 1 && o.scalar_i32(6, 1) == 1, "dilatation non gérée");
        (o.scalar_i8(0, 0) == 0, o.scalar_i32(2, 0) as usize, o.scalar_i32(1, 0) as usize, o.scalar_i8(4, 0))
    }

    /// FullyConnectedOptions : activation.
    pub fn fully_connected_activation(&self, op: &Operator) -> i8 {
        self.options(op).scalar_i8(0, 0)
    }
}

impl Tensor<'_> {
    pub fn as_f32(&self) -> Vec<f32> {
        self.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    pub fn as_i32(&self) -> Vec<i32> {
        self.data.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    pub fn as_i16_from_i8(&self) -> Vec<i16> {
        self.data.iter().map(|&b| b as i8 as i16).collect()
    }
}
